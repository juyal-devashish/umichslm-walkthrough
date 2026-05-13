"""Thin adapter that exposes ``UMichSLMForCausalLM`` with a torchtitan-style
forward signature.

The wrapper preserves the in-model loss fusion path (lm + mtp + moe_aux +
mod_aux) so the trainer can call ``output.loss.backward()`` directly. When
labels are provided the HF model returns ``logits=None`` (chunked loss path);
the wrapper mirrors this and exposes per-component scalars for logging.
"""

from __future__ import annotations

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
    def from_preset(cls, size: str, overrides: Optional[Dict[str, Any]] = None) -> "UMichSLMTitanWrapper":
        cfg_dict = _get_preset_dict(size).copy()
        if overrides:
            cfg_dict.update(overrides)
        config = UMichSLMConfig(**cfg_dict)
        inner = UMichSLMForCausalLM(config)
        return cls(inner)

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        tokens: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
    ):
        if labels is None:
            out = self.inner(
                input_ids=tokens,
                use_cache=False,
                return_dict=True,
            )
            return out.logits

        out = self.inner(
            input_ids=tokens,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return TitanModelOutput(
            loss=out.loss,
            lm_loss=getattr(out, "lm_loss", None),
            mtp_loss=getattr(out, "mtp_loss", None),
            aux_loss=getattr(out, "aux_loss", None),
        )

    # ----------------------------------------------------------- optimizations
    def apply_gradient_checkpointing(self) -> None:
        # The HF base class handles per-layer checkpointing under the hood;
        # this avoids monkey-patching individual decoder blocks.
        self.inner.gradient_checkpointing_enable()

    def apply_compile(self, *, deltanet_compile_safe: bool = True) -> int:
        """Per-block ``torch.compile`` on every decoder layer.

        Per-block (rather than whole-model) compile is friendlier to the FLA
        Triton kernels that DeltaNet calls into. When ``deltanet_compile_safe``
        is True, layers that route through DeltaNet are left in eager mode so
        a kernel-level Dynamo failure can't take down the run.
        Returns the number of layers actually compiled.
        """
        compiled = 0
        layers = self._iter_decoder_layers()
        use_deltanet = bool(getattr(self.inner.config, "use_deltanet", False))
        for idx, layer in enumerate(layers):
            if deltanet_compile_safe and use_deltanet and self._layer_uses_deltanet(idx):
                continue
            layer.forward = torch.compile(  # type: ignore[assignment]
                layer.forward, fullgraph=False, dynamic=False, mode="default"
            )
            compiled += 1
        return compiled

    def apply_float8(self) -> int:
        """Convert eligible ``nn.Linear`` layers to float8 training (Hopper+).

        Uses ``torchao.float8.convert_to_float8_training`` — Hopper's TC FP8
        path roughly doubles bf16 GEMM throughput for matmuls that meet the
        shape constraints (multiples of 16 / 32). We exclude the LM head and
        token embeddings: the head's `(hidden, vocab)` shape often fails the
        alignment check and the embeddings are not GEMMs, just lookups.

        Returns the number of layers actually converted. Raises ``ImportError``
        with a clear install hint when torchao isn't installed; raises
        ``RuntimeError`` on non-Hopper hardware (the kernel exists but isn't
        useful below SM 9.0). Called *before* ``apply_compile`` so the compiled
        graph captures the fp8 ops.
        """
        try:
            from torchao.float8 import (  # type: ignore[import-not-found]
                convert_to_float8_training,
            )
        except ImportError as exc:
            raise ImportError(
                "float8 training requires `torchao>=0.7`. "
                "Install with: pip install 'torchao>=0.7'"
            ) from exc

        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major < 9:
                raise RuntimeError(
                    f"float8 training needs SM>=9.0 (Hopper / Blackwell); "
                    f"got SM {major}.x. Set [training].float8 = false."
                )

        # Filter: only inner-block linears, skip lm_head and token embeddings.
        skip_keywords = ("lm_head", "embed_tokens", "wte", "wpe")

        def module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            return not any(kw in fqn for kw in skip_keywords)

        # Count first since the conversion mutates modules in place; reusing
        # the same predicate guarantees the count matches what was converted.
        eligible = [
            (fqn, m) for fqn, m in self.inner.named_modules()
            if module_filter(m, fqn)
        ]
        convert_to_float8_training(self.inner, module_filter_fn=module_filter)
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
        """Estimate per-token-active params (matters for MoE MFU).

        For dense models this equals ``num_params``. For MoE, we subtract the
        idle experts: each MoE block keeps only ``num_experts_per_tok`` of
        ``num_experts`` active per token.
        """
        cfg = self.inner.config
        if not getattr(cfg, "use_moe", False):
            return self.num_params

        num_experts = int(getattr(cfg, "num_experts", 1) or 1)
        active = int(getattr(cfg, "num_experts_per_tok", 1) or 1)
        if num_experts <= 1 or active >= num_experts:
            return self.num_params

        # Per-expert MLP params: 3 * hidden * intermediate (SwiGLU has 3 projections).
        moe_intermediate = int(
            getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size
        )
        per_expert = 3 * cfg.hidden_size * moe_intermediate
        idle_per_block = (num_experts - active) * per_expert
        idle_total = idle_per_block * cfg.num_hidden_layers
        return max(self.num_params - idle_total, per_expert)


def build_wrapper_from_config(model_cfg: ModelConfig, *, vocab_size: Optional[int] = None) -> UMichSLMTitanWrapper:
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

    if model_cfg.init_from:
        return UMichSLMTitanWrapper.from_pretrained(model_cfg.init_from, **overrides)

    return UMichSLMTitanWrapper.from_preset(model_cfg.size, overrides)
