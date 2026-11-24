"""Throughput instrumentation: step timing, FLOPs/token, MFU.

The headline deliverable is an apples-to-apples speed comparison vs LLaMA
Factory, so the FLOPs estimator and MFU calculation must be transparent and
configurable. Defaults follow Chinchilla-style accounting (``6 * N``) plus
attention quadratic, MoE active-experts adjustment, MTP head cost, and a
DeltaNet linear-attention term.
"""

from __future__ import annotations

import math
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

# ---------------------------------------------------------------- HW peaks
# Hardware peak bf16 dense TFLOP/s. Vendor-published except where a comment
# says otherwise; Float8 doubles the H100 figure. Prefer a MEASURED achievable
# peak where we have one -- a spec-sheet number no kernel reaches makes MFU
# read low for reasons that have nothing to do with the model.
HW_PEAK_TFLOPS: Dict[str, float] = {
    "h100_80gb_bf16": 989.0,
    "h100_80gb_fp8": 1979.0,
    # MEASURED, not a spec sheet: best-of-10 square bf16 GEMMs with fp32
    # accumulate, n = 4096 / 8192 / 16384 -> 378.5 / 390.6 / 388.8 TFLOP/s
    # (GL job 59164519, cu128). Flat across sizes, so this is an achievable
    # peak rather than a warm-up artefact. It replaced a 280.0 placeholder,
    # which means every Blackwell MFU reported before 2026-08-29 was
    # overstated by 390.6/280 = 1.395x.
    "rtx_pro_6000_blackwell_bf16": 390.6,
    "a100_80gb_bf16": 312.0,
    "rtx_6000_ada_bf16": 91.1,
    # RTX A6000 (Ampere GA102): 309.7 TFLOP/s bf16 tensor *with* 2:1 sparsity;
    # dense (what we actually hit) is half → 154.8. Distinct from "RTX 6000 Ada".
    "rtx_a6000_bf16": 154.8,
    # RTX A5000 (Ampere GA102, cut-down): 222.2 TFLOP/s bf16 tensor *with* 2:1
    # sparsity; dense is half → 111.1. Same halving as the A6000 above.
    "rtx_a5000_bf16": 111.1,
    # Great Lakes cards. Neither had an entry before 2026-10-21, so every spgpu /
    # spgpu2 run logged mfu_pct=0. MEASURED by greatlakes/gemm_peak.py (cu128,
    # best of 10, torch-default reduced-precision reductions, n = 4096 / 8192 /
    # 16384); the spec sheets say L40S 362.05 and A40 149.7 dense.
    # L40S (GL 64543896, gl1702): 256.6 / 237.3 / 212.7 -- 71% of spec, and FALLING
    # with size, i.e. the 350 W card power-throttles under sustained GEMM load.
    # A40 (GL 64543897, gl1519): 117.9 / 119.7 / 128.1 -- 86% of spec.
    "l40s_bf16": 256.6,
    "a40_bf16": 128.1,
}


def detect_hardware_key(precision: str = "bf16") -> str:
    """Best-effort GPU identifier. Falls back to a generic key."""
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0).lower()
    if "h100" in name:
        return "h100_80gb_fp8" if precision == "fp8" else "h100_80gb_bf16"
    if "rtx pro 6000" in name or "rtx 6000 blackwell" in name:
        return "rtx_pro_6000_blackwell_bf16"
    if "a100" in name:
        return "a100_80gb_bf16"
    if "rtx 6000" in name and "ada" in name:
        return "rtx_6000_ada_bf16"
    # Ampere "NVIDIA RTX A6000" (name contains "a6000", not the "rtx 6000 ada"
    # or "rtx pro 6000" strings matched above).
    if "a6000" in name:
        return "rtx_a6000_bf16"
    # Ampere "NVIDIA RTX A5000" (burger's calibration cards).
    if "a5000" in name:
        return "rtx_a5000_bf16"
    # Great Lakes: "NVIDIA L40S" (spgpu2) and "NVIDIA A40" (spgpu). Plain L40 is
    # half the L40S rate, so it must not match the L40S entry.
    if "l40s" in name:
        return "l40s_bf16"
    if re.search(r"\ba40\b", name):
        return "a40_bf16"
    return "unknown"


# -------------------------------------------------------------- step timer
class StepTimer:
    """CUDA-event-based step timer with peak-memory tracking.

    Use as ``timer.start()`` / ``timer.end()`` per training step. Times are
    measured on the default stream; results are float milliseconds.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._use_cuda = device.type == "cuda"
        if self._use_cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
        self._start_wall: Optional[float] = None

    def start(self) -> None:
        if self._use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            self._start.record()
        self._start_wall = time.perf_counter()

    def end(self) -> tuple[float, int]:
        wall_end = time.perf_counter()
        if self._use_cuda:
            self._end.record()
            torch.cuda.synchronize(self.device)
            elapsed_ms = float(self._start.elapsed_time(self._end))
            peak_bytes = int(torch.cuda.max_memory_allocated(self.device))
        else:
            elapsed_ms = (wall_end - (self._start_wall or wall_end)) * 1000.0
            peak_bytes = 0
        return elapsed_ms, peak_bytes


# ------------------------------------------------------------- flops counter
def avg_causal_keys(seq_len: int, window: Optional[int] = None) -> float:
    """Mean number of keys a query attends to, averaged over the sequence.

    Full causal (``window=None``): token t sees t+1 keys -> mean (S+1)/2.
    Sliding window W: token t sees min(t+1, W) -> mean
    ``[W(W+1)/2 + (S-W)W] / S`` for S > W, else full causal.

    This is THE correction that matters for the SWA family: the old model
    billed every layer as if it attended to all ``seq_len`` keys, which
    over-counted a 1024-window layer by 1.3x at 2048 and 2.3x at 4096 — i.e.
    it flattered exactly the architectures the sweep is screening.
    """
    s = float(seq_len)
    if window is None or window <= 0 or window >= seq_len:
        return (s + 1.0) / 2.0
    w = float(window)
    return (w * (w + 1.0) / 2.0 + (s - w) * w) / s


def per_layer_avg_keys(cfg, seq_len: int) -> List[float]:
    """Mean attended keys per layer, honoring the SWA band and any prefix.

    swa_globals_mla: ``"swa"`` entries use ``swa_window`` (plus the always-
    visible prefix globals, which sit outside the band); ``"mla"`` entries are
    full-context. Legacy configs use ``sliding_window`` on non-global layers.
    """
    from ..modules.sliding_window import is_global_attention_layer
    from ..modules.transformer_block import _should_use_deltanet

    num_layers = int(cfg.num_hidden_layers)
    mode = getattr(cfg, "attention_mode", "legacy")
    ratio = list(getattr(cfg, "hybrid_attention_ratio", None) or [])
    # Prefix globals are attended by every content token on top of the band;
    # the registers/sink twins keep the bank OUT of the sequence (billed in
    # the ``banks`` term instead), so they add nothing here.
    prefix = 0
    if mode == "swa_globals_mla" and not getattr(cfg, "swa_use_registers", False):
        prefix = int(getattr(cfg, "swa_global_count", 0) or 0)

    out: List[float] = []
    for idx in range(num_layers):
        entry = ratio[idx % len(ratio)] if ratio else None
        if _should_use_deltanet(cfg, idx):
            # No softmax attention: the recurrence is billed in the ``deltanet`` term.
            out.append(0.0)
        elif entry == "swa":
            window = int(getattr(cfg, "swa_window", 1024) or 1024)
            out.append(avg_causal_keys(seq_len, window) + prefix)
        elif entry in ("mla", "attn"):
            out.append(avg_causal_keys(seq_len) + prefix)
        else:
            sw = getattr(cfg, "sliding_window", None)
            every_n = getattr(cfg, "global_attention_every_n_layers", 4)
            if sw and not is_global_attention_layer(idx, every_n):
                out.append(avg_causal_keys(seq_len, int(sw)))
            else:
                out.append(avg_causal_keys(seq_len))
    return out


@dataclass
class FlopsBreakdown:
    base: float = 0.0
    attention: float = 0.0
    moe: float = 0.0
    mtp: float = 0.0
    deltanet: float = 0.0
    # Model-level dynamic compute (dyncompute chunk 2): GWT workspace trunk
    # and MoP pathway lanes. Reported separately from ``base`` because their
    # per-token cost is expectation-weighted, not param-proportional.
    trunk: float = 0.0
    lanes: float = 0.0
    # Out-of-sequence memory banks (sink / register / scribe reads). Their
    # attention work is invisible to ``base`` (the bank params are tiny) yet
    # real on the clock — omitting it understated MFU for every bank arm.
    banks: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.base
            + self.attention
            + self.moe
            + self.mtp
            + self.deltanet
            + self.trunk
            + self.lanes
            + self.banks
        )


class FlopsCounter:
    """Analytical per-token forward FLOPs estimator.

    Backward is approximated as 2x forward (Kaplan et al. convention), so the
    per-step training FLOPs reported is ``3 * forward_flops_per_token *
    tokens_per_step``.

    Notes:
        - "Active params" excludes idle MoE experts and discounts MoD-routed
          layers / multiplies recurrent-depth iterations (see
          ``UMichSLMTitanWrapper.num_active_params``). Plug that in via
          ``set_active_params``.
        - Attention quadratic term scales with seq_len^2 and is reported
          separately from the dense ``6*N*seq_len`` baseline so we can sanity
          check the breakdown in logs.
    """

    def __init__(self, model_config, seq_len: int):
        self.config = model_config
        self.seq_len = seq_len
        self._active_params: Optional[int] = None
        self._lane_cost_weights: Optional[Dict[str, float]] = None

    def set_active_params(self, n: int) -> None:
        self._active_params = int(n)

    def set_lane_cost_weights(self, weights: Optional[Dict[str, float]]) -> None:
        """Per-lane cost weighting for Mixture-of-Pathways.

        ``weights`` maps lane index (as a string, ``"0"``.. ) to the measured
        fraction of tokens routed to that lane (from the diagnostics panel).
        When set on a ``use_pathways`` config, the lanes term bills the
        realized expected lane cost instead of the ``pathway_budget`` default.
        Inert for non-pathway configs.
        """
        self._lane_cost_weights = dict(weights) if weights else None

    def per_token_forward_flops(self) -> FlopsBreakdown:
        cfg = self.config
        n_active = self._active_params or 0

        # Dense backbone term — ``2 * N`` FLOPs/token forward (Kaplan ``6N``
        # is forward+backward; we keep forward separate).
        base = 2.0 * n_active

        # Attention quadratic per token: ``2 * num_heads * head_dim * seq_len``
        # per layer (the seq_len^2 cost spread across tokens). Each layer is
        # weighted by its expected execution share: MoD-routed layers attend
        # only within the selected subset (capacity^2 scaling of the quadratic
        # term; dense under mod_mlp_only), and recurrent-depth core layers run
        # E[iterations] times. See umich_slm.compute_accounting.
        from ..compute_accounting import layer_compute_weights

        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
        # Per token per layer: 4 * d_model * (mean attended keys) — scores and
        # values, causal halving already inside ``avg_keys``. Windowed layers
        # bill their band, not the full sequence (see per_layer_avg_keys).
        d_model = cfg.num_attention_heads * head_dim
        weights = layer_compute_weights(cfg)
        avg_keys = per_layer_avg_keys(cfg, self.seq_len)
        # 4 * d_model is 2 * (score width + value width). MLA layers attend at their
        # own reconstructed widths (qk = nope + rope, v), not at hidden_size: 2.5x
        # hidden at the default 128/64/128 dims. The FA2 V zero-pad to the qk width
        # is hardware work, not model FLOPs, so it is not billed.
        ratio = list(getattr(cfg, "hybrid_attention_ratio", None) or [])
        mla_width = 2.0 * cfg.num_attention_heads * (
            int(getattr(cfg, "mla_qk_nope_head_dim", 128))
            + int(getattr(cfg, "mla_qk_rope_head_dim", 64))
            + int(getattr(cfg, "mla_v_head_dim", 128))
        )
        widths = [
            mla_width
            if (getattr(cfg, "use_mla", False) and ratio and ratio[i % len(ratio)] == "mla")
            else 4.0 * d_model
            for i in range(len(avg_keys))
        ]
        attn = sum(wd * w.attn_sq * k for wd, w, k in zip(widths, weights, avg_keys))

        # MoE term: ``2 * top_k * per_expert_FFN_FLOPs`` is already in ``base``
        # via active-params accounting. Nothing to add (kept here for clarity).
        moe = 0.0

        # MTP heads: ``mtp_num_heads * 2 * hidden * vocab`` per token (one
        # extra LM-head matmul per head). Hidden->vocab is the bulk of the
        # cost; the MTP transformer block per head is folded into ``base``.
        mtp = 0.0
        if getattr(cfg, "use_mtp", False):
            num_heads = int(getattr(cfg, "mtp_num_heads", 1) or 1)
            mtp = 2.0 * num_heads * cfg.hidden_size * cfg.vocab_size

        # Chunked gated delta rule, per token per head (estimate, chunk C=64):
        # state read q^T S (2*dk*dv) + delta-corrected write S(I - b k k^T) + b v k^T
        # (4*dk*dv) + intra-chunk causal scores/values and the WY solve
        # (~2*C*(dk+dv)). Billed ONLY on DeltaNet layers -- this used to multiply
        # by num_hidden_layers and also bill those layers as full softmax attention.
        deltanet = 0.0
        if getattr(cfg, "use_deltanet", False):
            from ..modules.transformer_block import _should_use_deltanet

            dn_heads = int(getattr(cfg, "deltanet_num_heads", None) or cfg.num_attention_heads)
            dn_head_dim = cfg.hidden_size // max(dn_heads, 1)
            dk = dn_head_dim * int(getattr(cfg, "deltanet_expand_k", 1) or 1)
            dv = dn_head_dim * int(getattr(cfg, "deltanet_expand_v", 1) or 1)
            # linear in tokens, so it scales like the projections (``attn``), not attn_sq
            n_dn = sum(w.attn for i, w in enumerate(weights) if _should_use_deltanet(cfg, i))
            chunk = 64.0
            deltanet = n_dn * dn_heads * (6.0 * dk * dv + 2.0 * chunk * (dk + dv))

        # GWT workspace trunk: param-proportional cost is already inside
        # ``base`` (num_active_params bills the amortized trunk), so this term
        # adds only the attention score/value FLOPs, same convention as
        # ``attn`` above (2 * width * context per query token):
        #   read (per token, context = slots), write (slots queries over the
        #   chunk, amortized), slot self-attention (slots^2 per chunk).
        trunk = 0.0
        if getattr(cfg, "use_workspace_trunk", False):
            from ..compute_accounting import expected_trunk_iters
            from ..modules.workspace_trunk import trunk_workspace_slots

            slots = float(trunk_workspace_slots(cfg))
            chunk = float(getattr(cfg, "trunk_chunk_size", 512))
            blocks = float(getattr(cfg, "trunk_layers", 2))
            e_iters = expected_trunk_iters(cfg)
            width = cfg.hidden_size
            trunk = 2.0 * width * slots * (1.0 + e_iters * blocks * (1.0 + slots / chunk))

        # MoP lanes: likewise only the cross-attention score/value FLOPs, per
        # executed lane block. Executed blocks come from measured lane fractions
        # when provided, else the pathway budget.
        lanes = 0.0
        if getattr(cfg, "use_pathways", False):
            from ..compute_accounting import pathway_expected_lane_blocks

            fractions = None
            if self._lane_cost_weights:
                depths = list(getattr(cfg, "pathway_depths", None) or [0, 2, 4])
                fractions = [
                    float(self._lane_cost_weights.get(str(i), 0.0)) for i in range(len(depths))
                ]
            executed = pathway_expected_lane_blocks(cfg, fractions)
            # Lane blocks are BANDED, not full-context: PathwayLaneBlock resolves
            # its window from pathway_lane_window, falling back to swa_window /
            # sliding_window (modules/pathways.py). Billing them at full causal
            # cost over-counts by 2.3x at 4096/W=1024, which lands as a ~5%
            # MFU over-report on the MoP arms only -- i.e. exactly the SWA-family
            # flattering that avg_causal_keys was introduced to remove.
            lane_window = getattr(cfg, "pathway_lane_window", None)
            if lane_window is None:
                lane_window = getattr(cfg, "swa_window", None) or getattr(
                    cfg, "sliding_window", None
                )
            lanes = (
                4.0
                * executed
                * d_model
                * avg_causal_keys(self.seq_len, int(lane_window) if lane_window else None)
            )

        # Out-of-sequence banks (swa_registers / swa_scribe family). Each read
        # is a cross-attention over a small key set on every SWA layer, plus
        # the extra q_proj/o_proj matmuls the read branch runs (the main path's
        # copies are already inside ``base``).
        banks = 0.0
        _ratio = list(getattr(cfg, "hybrid_attention_ratio", None) or [])
        # A looped core reads the bank once per ITERATION, so its SWA layers
        # count E[iters] times (2026-10-08: counting unique layers billed
        # swa_sinkdiet_gqa_rec at 0.979x its FLOP-matched parent). Only the
        # recurrence multiplier is applied, so every non-looped arm is unchanged.
        from ..compute_accounting import expected_recurrent_iters, recurrent_core_layers

        _core = set(recurrent_core_layers(cfg))
        _iters = expected_recurrent_iters(cfg) if _core else 1.0
        n_swa = float(
            sum(
                (_iters if i in _core else 1.0)
                for i in range(int(cfg.num_hidden_layers))
                if _ratio and _ratio[i % len(_ratio)] == "swa"
            )
        )
        if getattr(cfg, "swa_use_registers", False) and n_swa:
            n_reg = int(getattr(cfg, "swa_global_count", 0) or 0)
            reg_count = getattr(cfg, "swa_register_count", None)
            if reg_count is not None:
                n_reg = int(reg_count)
            # read attention + recomputed q_proj + the read's own o_proj
            banks += n_swa * (4.0 * d_model * n_reg + 4.0 * d_model * d_model)
        if int(getattr(cfg, "swa_sink_count", 0) or 0) > 0 and n_swa:
            banks += n_swa * 4.0 * d_model * int(cfg.swa_sink_count)
        if getattr(cfg, "use_scribe_bank", False) and n_swa:
            slots = int(getattr(cfg, "scribe_slots_per_chunk", 4) or 4)
            chunk = max(1, int(getattr(cfg, "scribe_chunk_size", 512) or 512))
            n_chunks = max(1.0, self.seq_len / chunk)
            # Read: a token in chunk c sees c*slots committed scribes; mean
            # over the sequence is slots*(C-1)/2. Only layers above the write
            # layer read, but bill all SWA layers (upper bound, few layers).
            banks += n_swa * 4.0 * d_model * slots * max(0.0, n_chunks - 1.0) / 2.0
            # Write (once, amortized per token): slots queries over the chunk
            # + the k/v projections of every chunk token.
            banks += 4.0 * d_model * slots + 4.0 * d_model * d_model
        if getattr(cfg, "use_delta_bank", False) and n_swa:
            # swa_deltabank. The READ is already billed by the registers branch
            # above (same N keys per token, per-chunk state instead of the
            # static bank — identical FLOPs), so only the WRITE is new: one
            # cross-attention per chunk with N queries over `chunk` tokens,
            # plus its q/k/v/o projections, amortized per token.
            n_reg = int(
                getattr(cfg, "swa_register_count", None) or getattr(cfg, "swa_global_count", 0) or 0
            )
            chunk = max(1, int(getattr(cfg, "delta_chunk_size", 512) or 512))
            # Per chunk: N queries x chunk keys (fwd 2 matmuls => 4*d*N*chunk
            # /chunk per token = 4*d*N) + N-query projections amortized over
            # the chunk + the chunk tokens' own k/v projections.
            banks += 4.0 * d_model * n_reg
            banks += (2.0 * 2.0 * d_model * d_model * n_reg) / chunk  # q,o on N queries
            banks += 2.0 * 2.0 * d_model * d_model  # k,v on every chunk token

        breakdown = FlopsBreakdown(
            base=base,
            attention=attn,
            moe=moe,
            mtp=mtp,
            deltanet=deltanet,
            trunk=trunk,
            lanes=lanes,
            banks=banks,
        )
        # Training-time compute bets re-run work the dense model does once, so
        # without this the reported MFU for these arms would be low by the whole
        # multiplier -- 28% for latent feedback, up to ~1.4x for the pause
        # stream. That is far outside the +-5% band attention-term corrections
        # live in, and MFU across arms is supposed to be readable (CLAUDE.md).
        return self._scale_for_extra_streams(breakdown, n_active)

    def _scale_for_extra_streams(self, b: "FlopsBreakdown", n_active: int):
        """Bill latent-feedback passes and the pause stream's second stream.

        LATENT FEEDBACK re-runs the entire stack -- embeddings, banks, pathway
        trunk and lm_head included, because every pass is supervised -- so
        every term scales by the EXPECTED pass count, 1.28x at the paper's
        default mixture. For ONE step this is approximate (the realized count
        is 1, 2 or 3); over a run it is exact. ``mtp`` is excluded because the
        MTP heads only run on the final pass.

        THE PAUSE STREAM is per-position work on a second residual stream, so
        everything per-position doubles -- attention, the register/sink bank
        read, the pathway trunk and lanes -- while two things do not:
          * the VOCABULARY TABLES. The input embedding is a lookup the
            prediction stream never does (it starts from e_pause), and lm_head
            reads the prediction stream only. At ar10m those tables are 63% of
            N, so getting this wrong would dominate the estimate.
          * the FFN, when ``pause_shared_ffn`` is on: it is evaluated once per
            position on a blend of the streams, not once per stream.
          * ``mtp``, which reads the final (prediction) hidden state at batch B.
        """
        import dataclasses

        cfg = self.config
        passes = 1.0
        if getattr(cfg, "use_latent_feedback", False):
            mix = getattr(cfg, "latent_feedback_pass_mix", None) or [1.0]
            passes = sum((k + 1) * float(w) for k, w in enumerate(mix))
        if passes != 1.0:
            b = dataclasses.replace(
                b,
                base=b.base * passes,
                attention=b.attention * passes,
                moe=b.moe * passes,
                deltanet=b.deltanet * passes,
                trunk=b.trunk * passes,
                lanes=b.lanes * passes,
                banks=b.banks * passes,
            )

        if not getattr(cfg, "use_pause_stream", False):
            return b

        h = int(cfg.hidden_size)
        tables = (
            int(getattr(cfg, "vocab_size", 0))
            * h
            * (1 if getattr(cfg, "tie_word_embeddings", False) else 2)
        )
        body = max(0.0, float(n_active) - float(tables))
        body_scale = 2.0
        if body > 0 and getattr(cfg, "pause_shared_ffn", True):
            ffn = (
                3.0 * h * int(getattr(cfg, "intermediate_size", 4 * h)) * int(cfg.num_hidden_layers)
            )
            ffn_share = min(1.0, ffn / body)
            body_scale = 2.0 * (1.0 - ffn_share) + 1.0 * ffn_share
        # b.base is 2*n_active: rescale only its body portion.
        base_scale = (
            (body * body_scale + (n_active - body)) / max(1.0, float(n_active)) if body > 0 else 1.0
        )
        return dataclasses.replace(
            b,
            base=b.base * base_scale,
            attention=b.attention * 2.0,
            moe=b.moe * 2.0,
            deltanet=b.deltanet * 2.0,
            trunk=b.trunk * 2.0,
            lanes=b.lanes * 2.0,
            banks=b.banks * 2.0,
        )

    def per_step_flops(self, *, tokens_per_step: int) -> float:
        """MODEL FLOPs per step (the MFU convention): 1x forward + 2x backward.

        Deliberately excludes activation-recompute — this matches PaLM-style
        MFU so numbers stay comparable to published figures. For what the
        silicon actually executes use :meth:`per_step_hardware_flops`.
        """
        forward = self.per_token_forward_flops().total * tokens_per_step
        return 3.0 * forward

    def per_step_hardware_flops(
        self, *, tokens_per_step: int, gradient_checkpointing: bool = False
    ) -> float:
        """HARDWARE FLOPs per step (the HFU convention).

        Full activation checkpointing re-runs the forward pass during backward,
        so the GPU executes 4x forward-equivalent while MFU credits 3x. With
        gradient checkpointing on — which is our default — HFU is 4/3 of MFU,
        i.e. reported MFU understates real utilization by ~25%.
        """
        forward = self.per_token_forward_flops().total * tokens_per_step
        return (4.0 if gradient_checkpointing else 3.0) * forward


# ------------------------------------------------------------------ MFU
class MFUTracker:
    def __init__(self, peak_tflops: float):
        self.peak_flops = peak_tflops * 1e12

    def mfu(self, *, total_flops: float, elapsed_seconds: float) -> float:
        if elapsed_seconds <= 0 or self.peak_flops <= 0:
            return 0.0
        return total_flops / (elapsed_seconds * self.peak_flops)


def gpu_inventory_snapshot() -> Dict[str, str]:
    """Capture ``nvidia-smi`` output once at run start for the comparison
    script's apples-to-apples assertion. Failures are non-fatal — the field
    is just informational metadata.
    """
    info: Dict[str, str] = {}
    if not torch.cuda.is_available():
        info["device"] = "cpu"
        return info
    info["device_name"] = torch.cuda.get_device_name(0)
    info["device_count"] = str(torch.cuda.device_count())
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode("utf-8")
            .strip()
        )
        info["nvidia_smi"] = out
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return info
