"""Thin adapter that exposes ``UMichSLMForCausalLM`` with a torchtitan-style
forward signature.

The wrapper preserves the in-model loss fusion path (lm + mtp + moe_aux +
mod_aux) so the trainer can call ``output.loss.backward()`` directly. When
labels are provided the HF model returns ``logits=None`` (chunked loss path);
the wrapper mirrors this and exposes per-component scalars for logging.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from ..configs.presets import _get_preset_dict
from ..configuration_umichslm import UMichSLMConfig
from ..modeling_umichslm import UMichSLMForCausalLM
from ..modules.transformer_block import UMichSLMDecoderLayer
from .config_schema import ModelConfig


@dataclass
class TitanModelOutput:
    """Lightweight forward output used in training mode.

    ``logits`` is intentionally omitted in the training path because the
    underlying model uses a chunked cross-entropy that never materializes the
    full ``(B, T, V)`` tensor (see ``modeling_umichslm.py:1131-1175``).
    """

    loss: torch.Tensor
    lm_loss: Optional[torch.Tensor] = None
    mtp_loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    # Detached per-token-loss stats (modules/agentic_heads.py) when the batch
    # carried ``token_targets``: asst_loss_sum/asst_count, errhead_loss/
    # errhead_count/errhead_scores, unlik_loss/unlik_count, lm_obj. None otherwise.
    token_stats: Optional[Dict[str, torch.Tensor]] = None


# Dynamo recompiles allowed per code frame. torch's default is 8; the gathered
# MoD path can legitimately need more (see apply_compile).
#
# Raised 32 -> 128 (2026-08-24). Every decoder block shares ONE code object
# (``TransformerBlock.forward``), so entries are consumed as
# (block variants x per-step specialization axes), not as max() of them. The
# dominant axis -- packed-document count, via ``_intra_doc_cu_seqlens`` -- is now
# removed at the source (``doc_masking._compile_friendly``), but arms differ in
# variant count (qwen3_like 1 graph, swa_sinkdiet 2, mat_ffn 5, think_suffix 11)
# and the MoD gather width still contributes up to 17 values, so the product can
# exceed 32 without any pathology. Headroom here is cheap -- one short compile per
# entry, once -- while the failure mode is silent eager execution for the rest of a
# multi-hour run, misread as an architecture being slow.
#
# Raised 128 -> 512 on 2026-08-29. The max_seqlen axis is now bounded at 32 values at
# ANY context (``doc_masking._MAX_SEQLEN_AXIS``) rather than ceil(S/128), but 32 is the
# per-axis bound, not the product: think_suffix's 11 block variants alone want 352
# entries and would still have blown a 128-entry cache at every context, 4096 included.
# Sized as (max variants 11) x (max_seqlen axis 32) with headroom. Job 59164519 is the
# case that forced it -- see doc_masking's note on the checkpoint crash it caused.
_DYNAMO_CACHE_SIZE = 512
# torch also caps the TOTAL recompiles across all frames (default 256). With 128
# per frame that ceiling would bind first and reintroduce the same silent fallback.
_DYNAMO_ACCUMULATED_LIMIT = 4096


class UMichSLMTitanWrapper(nn.Module):
    """Adapt the HF-style ``UMichSLMForCausalLM`` to a ``(tokens, labels)`` API.

    - ``forward(tokens, labels)`` returns a ``TitanModelOutput`` with the fused
      loss already computed by the inner model.
    - ``forward(tokens)`` (no labels) returns logits, for sampling/eval only.
    - ``apply_gradient_checkpointing()`` / ``apply_compile()`` configure the
      training-time optimizations torchtitan-style.
    - ``param_groups()`` returns weight-decay vs no-decay groups.
    """

    def __init__(self, inner: UMichSLMForCausalLM):
        super().__init__()
        self.inner = inner

    # ------------------------------------------------------------------ build
    @classmethod
    def from_pretrained(cls, path: str | Path, **overrides: Any) -> "UMichSLMTitanWrapper":
        inner = UMichSLMForCausalLM.from_pretrained(path, **overrides)
        return cls(inner)

    @classmethod
    def from_preset(
        cls, size: str, overrides: Optional[Dict[str, Any]] = None
    ) -> "UMichSLMTitanWrapper":
        cfg_dict = _get_preset_dict(size).copy()
        if overrides:
            cfg_dict.update(overrides)
        config = UMichSLMConfig(**cfg_dict)
        inner = UMichSLMForCausalLM(config)
        return cls(inner)

    def share_rope_caches(self) -> Tuple[int, int]:
        """Point rotary modules with byte-identical caches at one shared pair.

        Every decoder layer builds its OWN ``RotaryEmbedding``, so a model
        carries ``num_hidden_layers`` copies of the same fp32 cos/sin table.
        The tables scale with ``max_position_embeddings``, so this is a
        long-context tax specifically: at deep80 (L=80, head_dim=128) the
        duplicates cost 335 MB at 4096 and **2.68 GB at 32768** -- 6% of an
        A6000, competing directly with the activations that make long context
        hard in the first place.

        The caches are ``persistent=False`` buffers, so they never reach a
        state_dict and sharing them cannot interact with checkpointing (or
        with the safetensors tied-weight behaviour that once cold-restarted
        shared-lane resumes).

        Grouping is by signature AND verified byte-equality, so an arch that
        ever gives a layer its own theta -- a local/global split, say -- keeps
        a separate cache automatically instead of being silently flattened.

        MUST run after ``.to(device)``: ``nn.Module._apply`` copies each buffer
        independently, which would undo the sharing.
        """
        groups: Dict[Any, Any] = {}
        shared = 0
        saved = 0
        for mod in self.modules():
            cos = getattr(mod, "cos_cached", None)
            sin = getattr(mod, "sin_cached", None)
            if cos is None or sin is None:
                continue
            key = (
                tuple(cos.shape),
                cos.dtype,
                str(cos.device),
                float(getattr(mod, "base", 0.0) or 0.0),
                str(getattr(mod, "rope_type", "")),
            )
            canon = groups.get(key)
            if canon is None:
                groups[key] = (cos, sin)
                continue
            if not (torch.equal(canon[0], cos) and torch.equal(canon[1], sin)):
                continue
            saved += cos.numel() * cos.element_size() + sin.numel() * sin.element_size()
            mod.cos_cached = canon[0]
            mod.sin_cached = canon[1]
            shared += 1
        if shared:
            print(
                f"[titan] shared RoPE cos/sin caches across {shared} module(s), "
                f"freeing {saved / 1e9:.2f} GB",
                flush=True,
            )
        return shared, saved

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        tokens: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        document_ids: Optional[torch.LongTensor] = None,
        token_targets: Optional[Dict[str, Any]] = None,
    ):
        if labels is None:
            out = self.inner(
                input_ids=tokens,
                use_cache=False,
                return_dict=True,
                document_ids=document_ids,
            )
            return out.logits

        # token_targets is forwarded only when present, so the default call is
        # byte-identical to what it was before per-token losses existed.
        extra = {"token_targets": token_targets} if token_targets else {}
        self.inner._token_loss_stats = None
        out = self.inner(
            input_ids=tokens,
            labels=labels,
            use_cache=False,
            return_dict=True,
            document_ids=document_ids,
            **extra,
        )
        stats = getattr(self.inner, "_token_loss_stats", None)
        self.inner._token_loss_stats = None
        return TitanModelOutput(
            loss=out.loss,
            lm_loss=getattr(out, "lm_loss", None),
            mtp_loss=getattr(out, "mtp_loss", None),
            aux_loss=getattr(out, "aux_loss", None),
            token_stats=stats,
        )

    # ----------------------------------------------------------- optimizations
    def apply_gradient_checkpointing(self) -> None:
        # The HF base class handles per-layer checkpointing under the hood;
        # this avoids monkey-patching individual decoder blocks.
        self.inner.gradient_checkpointing_enable()

    def apply_compile(
        self,
        *,
        deltanet_compile_safe: bool = False,
        window_lse_compile_safe: bool = False,
        safe_pointwise_autotune: bool = True,
        compile_dyncompute_blocks: bool = True,
    ) -> int:
        """Per-block ``torch.compile`` on every decoder layer.

        Per-block (rather than whole-model) compile is friendlier to the FLA
        Triton kernels that DeltaNet calls into; FLA fences them itself
        (``torch.compiler.disable``), so a DeltaNet layer compiles as the fused
        graphs on either side of one opaque kernel call. When
        ``deltanet_compile_safe`` is True, layers that route through DeltaNet
        are left in eager mode instead (the pre-2026-10-20 default).
        Returns the number of layers actually compiled.

        ``safe_pointwise_autotune`` (default True) disables Inductor's pointwise
        autotuner, which otherwise benchmarks an over-reading config for the MLA
        flash V-slice kernel and crashes/NaNs the scribe/cmod arms on the A6000
        (see the inline comment for the full root cause). Throughput-neutral.
        ``window_lse_compile_safe`` is a blunter fallback for the same fault.

        ``suppress_errors`` makes a backend failure fall back to eager for the
        offending frame instead of raising. torch.compile is lazy — errors
        surface on the first forward, hours into a job — and the bank
        architectures do hit real backend bugs (inductor's CPU backend
        miscompiles the boolean bank masks: ``VecMask<long,2>``). Degrading to
        eager costs throughput; raising costs the whole run.

        ``cache_size_limit`` is raised from torch's default of 8 because
        ``dynamic=False`` specializes on shape and the block-atomic MoD runner
        feeds these same layers a GATHERED sequence whose width depends on the
        router. Fixed-capacity top-k gives one constant width (two shapes per
        layer, total). A THRESHOLD router (``mod_router_mode="threshold"``) meets
        capacity only in expectation, so the padded width wobbles step to step
        and each new value costs a recompile. Past the limit Dynamo abandons the
        frame and runs it eager FOREVER — and with ``suppress_errors`` on, it
        does so silently. That would read as "the causal-router arm is slower",
        i.e. as an architectural result, in a sweep whose whole purpose is to
        compare routing mechanisms. Widths span 2048..4096 in steps of
        ``mod_gather_bucket`` (128), so at most 17 values are reachable.

        That reasoning was right and the budget of 32 was still too small
        (2026-08-24): entries are consumed as (block variants x axes), because
        every block shares one ``TransformerBlock.forward`` code object. The
        dominant axis was packed-document count, and it is now removed at the
        source in ``doc_masking._compile_friendly``; the limit is 128 for the
        remainder. Each extra entry costs one short compile, once — no recurring
        compute, unlike widening the bucket.
        """
        torch._dynamo.config.suppress_errors = True
        # Env override exists for IN-JOB A/B against the historical limit of 32
        # (see doc_masking._compile_friendly's escape hatch); not used in training.
        cache_size = _DYNAMO_CACHE_SIZE
        _env = os.environ.get("UMICHSLM_DYNAMO_CACHE_SIZE")
        if _env:
            try:
                cache_size = int(_env)
            except ValueError:
                pass
        # torch >= 2.9 renamed cache_size_limit -> recompile_limit and keeps the old
        # name as an alias; set whichever exist so the limit lands on every version.
        for _name in ("cache_size_limit", "recompile_limit"):
            if getattr(torch._dynamo.config, _name, None) is not None:
                if getattr(torch._dynamo.config, _name) != cache_size:
                    setattr(torch._dynamo.config, _name, cache_size)
        for _name in ("accumulated_cache_size_limit", "accumulated_recompile_limit"):
            if getattr(torch._dynamo.config, _name, None) is not None:
                if getattr(torch._dynamo.config, _name) < _DYNAMO_ACCUMULATED_LIMIT:
                    setattr(torch._dynamo.config, _name, _DYNAMO_ACCUMULATED_LIMIT)
        if safe_pointwise_autotune:
            # Minor throughput precaution, NOT a correctness fix. The scribe/cmod
            # crashes once attributed here (an autotuned pointwise over-read of the
            # MLA V-pad slice) were MISDIAGNOSED: the real fault was the MoD gather
            # runner feeding flash_attn_varlen a full-sequence cu_seqlens against a
            # compact [B*k, ...] buffer -- fixed at the source in
            # UMichSLMModel._run_gathered_mod_block (forces compact members onto
            # SDPA). That crash is flag-independent (it reproduces identically with
            # autotune on OR off, in eager OR compile, on A5000 OR A6000 with real
            # packed data). We keep pointwise autotune OFF only because it measured
            # marginally faster (~+0.3k tok/s: pointwise kernels are memory-bound,
            # so benchmarking XBLOCK/warp configs rarely helps) and leaves the
            # matmul/reduction autotuners (what MFU depends on) untouched. Safe to
            # flip back on with no correctness consequence.
            try:
                import torch._inductor.config as _inductor_config

                _inductor_config.triton.autotune_pointwise = False
            except Exception as e:  # inductor internals may move across versions
                # Do NOT swallow silently: this flag is the documented throughput
                # precaution for the MLA V-slice path, and a moved Inductor API
                # would disable it with no signal. Not a correctness issue (the OOB
                # was fixed at source), but surface it so the API move is noticed.
                warnings.warn(
                    "apply_compile: could not disable Inductor pointwise autotune "
                    f"({type(e).__name__}: {e}); the MLA V-slice throughput "
                    "precaution is inactive -- the Inductor API may have moved, "
                    "update wrapper.apply_compile.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        compiled = 0
        layers = self._iter_decoder_layers()
        cfg = self.inner.config
        use_deltanet = bool(getattr(cfg, "use_deltanet", False))
        # Fallback escape hatch (default OFF; safe_pointwise_autotune is the real
        # fix). When enabled, leaves the window-LSE (sink/scribe) layers eager for
        # the MoD gather-runner arms (use_mod && !use_pathways) -- a blunter way to
        # dodge the same Inductor fault at a ~17% throughput cost on those arms.
        lse_gather_arm = bool(getattr(cfg, "use_mod", False)) and not bool(
            getattr(cfg, "use_pathways", False)
        )
        lse_skipped = 0
        for idx, layer in enumerate(layers):
            if deltanet_compile_safe and use_deltanet and self._layer_uses_deltanet(idx):
                continue
            if window_lse_compile_safe and lse_gather_arm and self._layer_wants_window_lse(layer):
                lse_skipped += 1
                continue
            layer.forward = torch.compile(  # type: ignore[assignment]
                layer.forward, fullgraph=False, dynamic=False, mode="default"
            )
            compiled += 1
        if lse_skipped:
            print(
                f"[titan] window_lse_compile_safe: left {lse_skipped} window-LSE "
                "(sink/scribe) layers eager under the MoD gather runner "
                "(Inductor codegen fault on A6000); MLA global layers stay compiled."
            )
        if compile_dyncompute_blocks:
            compiled += self._compile_dyncompute_blocks()
        return compiled

    def _compile_dyncompute_blocks(self) -> int:
        """Per-block compile for the dynamic-compute modules that run OUTSIDE
        the decoder layers (2026-08-18 mopd deep-dive): MoP lane blocks and the
        batched workspace-trunk variants' blocks. These were the eager-dispatch
        hot spots — mopd spent ~57 ms/step of HOST time per Pathways stage on
        eager launches while the GPU was only ~55% busy. Lane blocks see a
        handful of bucketed gather widths (covered by the raised
        cache_size_limit); batched trunk blocks see exactly one shape. The
        SEQUENTIAL trunk stays eager — its remedy is trunk_cuda_graph (gwt) or
        a batched trunk_variant, not per-block compile of a 7-round serial
        chain. suppress_errors (set by apply_compile) keeps any backend failure
        a throughput event, never a crash."""
        compiled = 0
        base = getattr(self.inner, "model", None)
        if base is None:
            return 0
        cfg = self.inner.config
        stages = []
        if getattr(base, "pathway_stages", None) is not None:
            stages = list(base.pathway_stages)
        elif getattr(base, "pathways", None) is not None:
            stages = [base.pathways]
        seen: set = set()
        _prologue_on = os.environ.get("UMICH_COMPILE_STAGE_PROLOGUE", "1") != "0"
        # Whole-stage fusion (experimental, default OFF): compile the entire
        # Pathways.forward per stage with fullgraph=False — dynamo splits at
        # the tier-count host sync and mask builds, but the gather->lane and
        # epilogue->scatter chains land INSIDE compiled subgraphs so Inductor
        # can restructure the bandwidth-bound memory traffic (the residual
        # ~250 ms/step the piecewise merge compile could not recover). When
        # on, the per-piece compiles below are skipped: nesting compiled
        # regions would double guard overhead and re-trace the inner wrappers.
        _full_on = os.environ.get("UMICH_COMPILE_STAGE_FULL", "0") == "1"
        if _full_on:
            for stage in stages:
                stage.forward = torch.compile(  # type: ignore[assignment]
                    stage.forward, fullgraph=False, dynamic=False, mode="default"
                )
                compiled += 1
            stages = []
        for stage in stages:
            if _prologue_on and getattr(stage, "export_interval", 1) <= 1:
                # Stage prologue (export norm + shared K/V + router): one
                # static-shape graph replacing ~30 eager launches per stage —
                # about half the measured ~32 ms/stage fixed machinery
                # (round-3 attribution). All stages share one code object, so
                # dynamo compiles once and reuses across the 11-21 stages.
                # Skipped under pathway_export_interval > 1: that path routes
                # through the export cache in eager forward instead.
                stage._stage_prologue = torch.compile(  # type: ignore[assignment]
                    stage._stage_prologue, fullgraph=False, dynamic=False, mode="default"
                )
                compiled += 1
            if os.environ.get("UMICH_COMPILE_STAGE_MERGE", "1") != "0":
                # Merge epilogue: the round-4 profile put ~250 ms/step of the
                # 11-stage mopr cell in the BACKWARD of the eager merge chains
                # (gather -> norm -> scale -> gate -> scatter_add, full-seq
                # intermediates per stage). Compiling fuses fwd AND generated
                # bwd; gathered widths are bucketed so variants stay few.
                stage._merge_dense = torch.compile(  # type: ignore[assignment]
                    stage._merge_dense, fullgraph=False, dynamic=False, mode="default"
                )
                stage._merge_gathered = torch.compile(  # type: ignore[assignment]
                    stage._merge_gathered, fullgraph=False, dynamic=False, mode="default"
                )
                compiled += 1
            for block in stage.iter_lane_blocks():
                if id(block) in seen:  # shared/recursive lanes repeat objects
                    continue
                seen.add(id(block))
                block.forward = torch.compile(  # type: ignore[assignment]
                    block.forward, fullgraph=False, dynamic=False, mode="default"
                )
                compiled += 1
            mem_block = getattr(stage, "memory_block", None)
            if mem_block is not None and id(mem_block) not in seen:
                seen.add(id(mem_block))
                mem_block.forward = torch.compile(  # type: ignore[assignment]
                    mem_block.forward, fullgraph=False, dynamic=False, mode="default"
                )
                compiled += 1
        mixer = getattr(base, "recurrent_mixer", None)
        if mixer is not None:
            # Loop-boundary mixers run at model level, outside the per-layer
            # compile, once per iteration: a dozen small elementwise kernels plus
            # one GEMM each, which eager would dispatch one by one. One shape per
            # run; each hyper-connection set is its own module (one entry each).
            if mixer.injection is not None:
                mixer.injection.forward = torch.compile(  # type: ignore[assignment]
                    mixer.injection.forward, fullgraph=False, dynamic=False, mode="default"
                )
                compiled += 1
            for hc in mixer.hyper or []:
                hc.begin = torch.compile(  # type: ignore[assignment]
                    hc.begin, fullgraph=False, dynamic=False, mode="default"
                )
                hc.end = torch.compile(  # type: ignore[assignment]
                    hc.end, fullgraph=False, dynamic=False, mode="default"
                )
                compiled += 1
        trunk = getattr(base, "workspace_trunk", None)
        if trunk is not None and str(getattr(cfg, "trunk_variant", "sequential")) != "sequential":
            tseen: set = set()
            for block in trunk.blocks:
                if id(block) in tseen:  # weight-shared trunks repeat objects
                    continue
                tseen.add(id(block))
                block.forward = torch.compile(  # type: ignore[assignment]
                    block.forward, fullgraph=False, dynamic=False, mode="default"
                )
                compiled += 1
            if hasattr(trunk, "_scan_update"):
                # Gated-scan variants (rglru / delta_rule): the chunk-axis scan
                # is a fixed-length chain of tiny elementwise kernels — 7 iters
                # at 4096/512 but 63 at 32K, where eager dispatch cost ~250 ms
                # over partrunk (2026-08-18 A5000). Compile unrolls and fuses
                # it; one static shape per (B, S).
                trunk._reader_states = torch.compile(  # type: ignore[assignment]
                    trunk._reader_states, fullgraph=False, dynamic=False, mode="default"
                )
                compiled += 1
        mems = getattr(base, "delta_memories", None)
        if mems is not None:
            # DeltaMemory runs BETWEEN decoder layers, at model level, so the
            # per-layer compile above never reached it and it ran fully eager --
            # measured in gate T (2026-08-21) as 7530 ms/step at B=8 against a
            # 1561 ms dense anchor, i.e. 4.8x, with median/min spread of 23%,
            # the signature of a host-bound dispatch loop rather than an
            # arithmetic cost. The chunk loop is S/chunk sequential iterations
            # of ~10 tiny kernels each, on [B,H,chunk,D] with H=4 and D=64;
            # at S=4096, chunk=64 and interval=4 that is ~13k launches per
            # forward and again per grad-ckpt recompute. Every module here
            # sees exactly one input shape, so this is one graph reused across
            # all of them.
            for mem in mems:
                if id(mem) in seen:
                    continue
                seen.add(id(mem))
                mem.forward = torch.compile(  # type: ignore[assignment]
                    mem.forward, fullgraph=False, dynamic=False, mode="default"
                )
                compiled += 1
        return compiled

    def apply_float8(self, recipe: Optional[str] = None) -> int:
        """Convert eligible ``nn.Linear`` layers to float8 training (Hopper+).

        Uses ``torchao.float8.convert_to_float8_training`` — Hopper's TC FP8
        path roughly doubles bf16 GEMM throughput for matmuls that meet the
        shape constraints (multiples of 16 / 32). We exclude the LM head and
        token embeddings: the head's `(hidden, vocab)` shape often fails the
        alignment check and the embeddings are not GEMMs, just lookups. Linears
        whose dims are not multiples of 16 (the routing heads) are excluded
        too -- torch._scaled_mm rejects them outright.

        Returns the number of layers actually converted. Raises ``ImportError``
        with a clear install hint when torchao isn't installed; raises
        ``RuntimeError`` on non-Hopper hardware (the kernel exists but isn't
        useful below SM 9.0). Called *before* ``apply_compile`` so the compiled
        graph captures the fp8 ops.
        """
        try:
            from torchao.float8 import (  # type: ignore[import-not-found]
                Float8LinearConfig,
                convert_to_float8_training,
            )
        except ImportError as exc:
            raise ImportError(
                "float8 training requires `torchao>=0.7`. "
                "Install with: pip install 'torchao>=0.7'"
            ) from exc

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            # The real floor is torch._scaled_mm's, which is sm_89 -- NOT sm_90.
            # `major < 9` also rejected sm_89 (L40S / RTX 4090), which support
            # FP8 tensor cores perfectly well. Compare the full (major, minor).
            # Cards in reach and where they land: A100 sm_80 and A6000/A40 sm_86
            # are genuinely below the floor (no local card can run FP8 at all);
            # L40S sm_89, H100 sm_90 and RTX PRO 6000 Blackwell sm_120 are above.
            if (major, minor) < (8, 9):
                raise RuntimeError(
                    f"float8 training needs SM>=8.9 (Ada / Hopper / Blackwell); "
                    f"got SM {major}.{minor}. Set [training].float8 = false."
                )

        # Filter: only inner-block linears, skip lm_head and token embeddings.
        skip_keywords = ("lm_head", "embed_tokens", "wte", "wpe")

        def module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if any(kw in fqn for kw in skip_keywords):
                return False
            # torch._scaled_mm requires BOTH GEMM dims to be multiples of 16;
            # a name-only filter silently admits the tiny routing heads and the
            # run dies at the first forward with
            #   RuntimeError: mat2 shape (896x4) must be divisible by 16
            # That is not hypothetical -- it is what blocked FP8 for
            # mat_ffn_thresh_gqa (4-tier router) and think_suffix_gqa (recurse
            # router) on both recipes and both CUDA builds in Great Lakes job
            # 58822457, while the router-free arms ran fine. These heads are a
            # rounding error in FLOPs, so excluding them costs nothing.
            return mod.in_features % 16 == 0 and mod.out_features % 16 == 0

        # Count first since the conversion mutates modules in place; reusing
        # the same predicate guarantees the count matches what was converted.
        eligible = [(fqn, m) for fqn, m in self.inner.named_modules() if module_filter(m, fqn)]
        unaligned = [
            f"{fqn}({m.in_features}x{m.out_features})"
            for fqn, m in self.inner.named_modules()
            if isinstance(m, nn.Linear)
            and not any(kw in fqn for kw in skip_keywords)
            and (m.in_features % 16 or m.out_features % 16)
        ]
        if unaligned:
            print(
                f"[titan] float8: skipping {len(unaligned)} unaligned linear(s), "
                f"e.g. {', '.join(sorted(set(unaligned))[:3])}",
                flush=True,
            )
        # recipe: torchao's default is tensorwise dynamic scaling (one scale per
        # tensor). "rowwise" keeps a scale per row, which costs a little
        # throughput and buys back most of the accuracy tensorwise loses on
        # outlier-heavy activations -- the reason most production FP8 pretraining
        # now uses it. Passing None keeps torchao's default so existing configs
        # are unchanged.
        cfg = Float8LinearConfig.from_recipe_name(recipe) if recipe else None
        convert_to_float8_training(self.inner, module_filter_fn=module_filter, config=cfg)
        return len(eligible)

    def _iter_decoder_layers(self) -> List[UMichSLMDecoderLayer]:
        # The base model is at ``self.inner.model``; its ``layers`` attribute is
        # an ``nn.ModuleList`` of decoder blocks. Tolerate model variants that
        # nest the layers slightly differently.
        base = getattr(self.inner, "model", None)
        if base is None or not hasattr(base, "layers"):
            return []
        return list(base.layers)

    def _layer_uses_deltanet(self, layer_idx: int) -> bool:
        cfg = self.inner.config
        pattern = getattr(cfg, "hybrid_attention_ratio", None)
        if pattern:
            return pattern[layer_idx % len(pattern)] == "delta"
        first_n = int(getattr(cfg, "deltanet_first_n_layers", 0) or 0)
        if layer_idx < first_n:
            return False
        every_n = int(getattr(cfg, "deltanet_every_n_layers", 0) or 0)
        if every_n <= 0:
            return True
        adjusted = layer_idx - first_n
        return (adjusted + 1) % every_n != 0

    @staticmethod
    def _layer_wants_window_lse(layer) -> bool:
        """True if the block contains a window-LSE (sink/scribe) attention.

        ``_wants_window_lse`` is set on the scribe/sink attention module
        (``modules/swa_scribe.py``); read it off the live submodules rather than
        re-deriving from config, per CLAUDE.md ("confirm the knob reached the
        model").
        """
        return any(getattr(m, "_wants_window_lse", False) for m in layer.modules())

    # ---------------------------------------------------------- pause phasing
    @property
    def has_pause_stream(self) -> bool:
        """True when the inner model carries a ``_pause`` prediction stream.

        train.py's phasing used to probe ``wrapper.model.pause_stream``; the
        wrapper has no ``.model`` (the HF model is ``self.inner``), so the probe
        read None, ``pause_phase_start_frac`` was silently ignored, and every
        ``*_phase75`` cell before 2026-10-26 trained full ``_pause`` from step 0
        (flat step time across the switch, no ``[pause]`` log line).
        """
        return getattr(self.inner.model, "pause_stream", None) is not None

    def set_pause_enabled(self, enabled: bool) -> bool:
        """Delegate to ``UMichSLMModel.set_pause_enabled``; returns the flag set."""
        return self.inner.model.set_pause_enabled(enabled)

    @property
    def has_mtp(self) -> bool:
        """True when the inner model carries MTP heads (``use_mtp``)."""
        return getattr(self.inner, "mtp", None) is not None

    def set_mtp_enabled(self, enabled: bool) -> bool:
        """Delegate to ``UMichSLMForCausalLM.set_mtp_enabled``; returns the flag set."""
        return self.inner.set_mtp_enabled(enabled)

    # --------------------------------------------------------- optimizer setup
    def param_groups(self, weight_decay: float) -> List[Dict[str, Any]]:
        """Split parameters into weight-decay and no-decay groups.

        Excludes biases and norm weights from decay (matches the convention
        used in ``scripts/pretrain_fineweb.py`` and most LLM training recipes).
        """
        decay: List[nn.Parameter] = []
        no_decay: List[nn.Parameter] = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # --------------------------------------------------------- introspection
    @property
    def config(self) -> UMichSLMConfig:
        return self.inner.config

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_active_params(self) -> int:
        """Estimate per-token-active params (drives FLOPs/MFU accounting).

        For dense models this equals ``num_params``. Three adjustments:

        - **MoE**: subtract idle experts (each block keeps only
          ``num_experts_per_tok`` of ``num_experts`` active per token).
        - **MoD / MoD-MLP**: routed layers bill at their routing capacity
          (FFN-only under ``mod_mlp_only``) — see
          ``umich_slm.compute_accounting``.
        - **Recurrent depth / weight sharing**: layer *executions* are billed,
          so a looped or weight-shared block counts once per pass even though
          its params are deduped in ``num_params``.

        The accounting is algorithmic (what the arch fundamentally computes),
        not realized wall-clock — ``submit_arch_mfu.sh`` is the ground truth
        for the latter.
        """
        from ..compute_accounting import layer_compute_param_delta, model_compute_param_delta

        cfg = self.inner.config
        n = float(self.num_params)

        if getattr(cfg, "use_moe", False):
            num_experts = int(getattr(cfg, "num_experts", 1) or 1)
            active = int(getattr(cfg, "num_experts_per_tok", 1) or 1)
            if num_experts > 1 and active < num_experts:
                # Per-expert MLP params: 3 * hidden * intermediate (SwiGLU).
                moe_intermediate = int(
                    getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size
                )
                per_expert = 3 * cfg.hidden_size * moe_intermediate
                idle_total = (num_experts - active) * per_expert * cfg.num_hidden_layers
                n = max(n - idle_total, float(per_expert))

        base = getattr(self.inner, "model", None)
        if base is not None and hasattr(base, "layers"):
            # Layer routing/recursion + model-level dynamic compute (GWT
            # workspace trunk, MoP pathway lanes).
            n += model_compute_param_delta(base, cfg)
        else:
            n += layer_compute_param_delta(self._iter_decoder_layers(), cfg)
        return max(int(round(n)), 1)


def fit_rope_cache_to_seq(overrides: Dict[str, Any], preset: str, seq_len: int) -> Optional[int]:
    """Size ``max_position_embeddings`` to the sequence actually being trained.

    Presets ship ``max_position_embeddings=4096`` and nothing downstream ever
    derived it from ``[data].seq_len`` / ``--seq``. That is not a benign
    mismatch: ``rotary.py`` builds the cos/sin cache to
    ``max_position_embeddings`` at construction and ``_maybe_extend_cache``
    deliberately skips the grow-check under ``torch.compile`` (reading
    ``position_ids.max()`` is a host sync and a hard graph break). So a
    compiled run at S > max_position_embeddings indexes ``cos_cached`` out of
    bounds and dies with an async CUDA device-side assert -- which is exactly
    what killed every S>=8192 cell of Great Lakes job 58953441 on all three
    architectures while S=4096 passed.

    For ``rope_type`` "default" the value is *only* a cache length: no
    parameters, no FLOPs, and identical math for every position below the old
    limit. Fitting it to the run is therefore the correct behaviour and it is
    logged rather than silent.

    For yarn/dynamic/linear scaling it is load-bearing -- ``factor`` is defined
    relative to it -- so we refuse to guess and ask for an explicit value.
    The one exception is yarn with ``rope_scaling.original_max_position_embeddings``
    set: the base length is then carried there (HF convention, read by
    ``RotaryEmbedding``), and ``max_position_embeddings`` is a cache length again.
    An explicit ``max_position_embeddings`` override always wins.
    """
    if "max_position_embeddings" in overrides:
        return None
    base = dict(_get_preset_dict(preset))
    base.update(overrides)
    current = int(base.get("max_position_embeddings", 0) or 0)
    if seq_len <= current:
        return None
    scaling = base.get("rope_scaling") or {}
    rope_type = str(scaling.get("rope_type") or scaling.get("type") or "default")
    base_carried = rope_type == "yarn" and scaling.get("original_max_position_embeddings")
    if rope_type not in ("default", "none") and not base_carried:
        # Do NOT "keep the base length in max_position_embeddings" for YaRN: the
        # cos/sin cache is sized from it and a compiled run never grows the cache,
        # so every position past the base length would index out of bounds.
        raise ValueError(
            f"seq_len={seq_len} exceeds max_position_embeddings={current} and "
            f"rope_scaling.rope_type={rope_type!r} makes that value load-bearing "
            "(the scaling factor is defined relative to it). For a YaRN extension set "
            "max_position_embeddings to the NEW length and "
            "rope_scaling.original_max_position_embeddings to the BASE length."
        )
    overrides["max_position_embeddings"] = seq_len
    print(
        f"[titan] max_position_embeddings {current} -> {seq_len} to cover "
        f"seq_len={seq_len} (rope_type={rope_type}: cache length only, no param "
        "or FLOP change)",
        flush=True,
    )
    return seq_len


def resolve_attn_implementation(explicit: Optional[str]) -> str:
    """Pick the attention backend for a live training run.

    An explicit setting always wins. Otherwise this walks the package's
    documented ladder -- flash_attention_2 -> sdpa -- instead of leaving the
    choice to HF, whose default is sdpa.

    WHY THIS EXISTS. ``bench_dyncompute.py`` has always defaulted to
    ``flash_attention_2`` while this path left ``attn_implementation`` at None,
    so the benched configuration and the trained one differed in their
    attention kernel and nobody had a reason to look. Great Lakes job 58992621
    was built as a 2x2 (impl x TORCHINDUCTOR_FORCE_DISABLE_CACHES) to find out
    which axis mattered, and the answer was unambiguous: both flash cells
    passed, both sdpa cells failed identically, cache setting irrelevant::

        assert_size_stride(buf23, (16, 7, 4096, 128), (3670016, 524288, 128, 1),
            'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        AssertionError: expected size 7==7, stride 128==524288 at dim=1;
                        expected size 4096==4096, stride 896==128 at dim=2

    i.e. Inductor guards that op for a [B,S,H,D]-contiguous layout and we hand
    it [B,H,S,D]. So on Blackwell every *compiled live* run died while every
    bench cell at the same shapes passed. This is also the assertion previously
    blamed on the cu130 environment; job 58953442 had already reproduced it on
    cu128 at 1 and 2 ranks, and this 2x2 identifies the real axis as the
    attention path. (The separate cu130 bf16-forward miscompile is unrelated
    and still stands.)

    Falls back to sdpa when flash-attn is not importable -- on a card where the
    bug bites, a run that will not compile is worse than a slower kernel, but
    that is a fallback, not the default.
    """
    if explicit:
        return explicit
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return "sdpa"
    return "flash_attention_2"


def build_wrapper_from_config(
    model_cfg: ModelConfig,
    *,
    vocab_size: Optional[int] = None,
    seq_len: Optional[int] = None,
) -> UMichSLMTitanWrapper:
    """Construct a wrapper from a parsed ``[model]`` section.

    If ``init_from`` is provided, the underlying HF model is loaded from that
    path (must be a directory written by ``save_pretrained``). Otherwise the
    preset is instantiated with optional overrides. ``vocab_size`` overrides
    the preset placeholder when supplied (this is the contract: presets ship
    a placeholder vocab and the trainer must set it from the tokenizer).
    """
    overrides = dict(model_cfg.overrides or {})
    if vocab_size is not None:
        overrides.setdefault("vocab_size", vocab_size)
    # PretrainedConfig consumes the `attn_implementation` kwarg (-> config._attn_
    # implementation), which both the preset and from_pretrained paths honour.
    explicit = getattr(model_cfg, "attn_implementation", None)
    impl = resolve_attn_implementation(explicit)
    if "attn_implementation" not in overrides:
        overrides["attn_implementation"] = impl
        if not explicit:
            print(f"[titan] attn_implementation -> {impl!r} (ladder default)", flush=True)
    if seq_len is not None and not model_cfg.init_from:
        fit_rope_cache_to_seq(overrides, model_cfg.size, int(seq_len))

    if model_cfg.init_from:
        return UMichSLMTitanWrapper.from_pretrained(model_cfg.init_from, **overrides)

    return UMichSLMTitanWrapper.from_preset(model_cfg.size, overrides)
