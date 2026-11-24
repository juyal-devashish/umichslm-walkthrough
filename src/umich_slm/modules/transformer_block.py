"""Transformer decoder layer implementation.

This module implements a complete transformer decoder layer that combines
attention, MLP, and normalization components in a pre-norm architecture.
Includes support for Mixture of Experts (MoE) layers.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .attention import ATTENTION_CLASSES, UMichSLMAttention
from .deltanet import DELTANET_CLASSES, GatedDeltaNet
from .griffin import GRIFFIN_CLASSES, RGLRUAttention
from .mla import MLA_ATTENTION_CLASSES, UMichSLMMLAAttention
from .moe import MoELayer, MoEOutput, SparseMoE
from .rmsnorm import get_rmsnorm_class
from .swiglu import MatFormerSwiGLUMLP, RoutedMatFormerSwiGLUMLP, SwiGLUMLP


def _build_mlp(config, layer_idx=None):
    """Build the per-block FFN: SwiGLU, or MatFormer nested-FFN when use_matformer.

    FIXED-COMPUTE PREFIX/SUFFIX. Per the design docs, the first few and last few
    layers must run FULL compute for every token -- early layers are still
    building the representation the router would have to judge, and a token
    starved there cannot recover. Only the middle "reasoning" layers are
    dynamic. `mat_ffn` therefore keeps a plain full-width SwiGLU outside
    [dyn_start, L - dyn_end), matching the mod_num_always_used_start/_end
    convention MoD already uses.
    """
    kinds = getattr(config, "mat_ffn_layer_kinds", None)
    if (
        getattr(config, "use_mat_ffn", False)
        and layer_idx is not None
        and kinds
        and _ratio_entry_for_layer(config, layer_idx) not in kinds
    ):
        # mat_ffn_layer_kinds: this layer's kind is not routed. Plain SwiGLU at the
        # backbone width, which is what mat_ffn_fixed_width holds (intermediate_size
        # is the nest's WIDEST tier).
        fw = getattr(config, "mat_ffn_fixed_width", None)
        return SwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=int(fw) if fw else config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            bias=getattr(config, "mlp_bias", False),
        )
    if getattr(config, "use_mat_ffn", False) and layer_idx is not None:
        L = int(config.num_hidden_layers)
        # CLAMP the fixed prefix/suffix to the model's depth. With the defaults
        # (8 and 8) a 12-layer model would get the band [8, 4) -- EMPTY -- and
        # mat_ffn would silently become a plain dense model with no routers at
        # all, which is exactly the class of bug this codebase keeps hitting.
        # Deep models are unaffected (at L=80 the band is [8, 72) as intended);
        # shallow ones keep a proportional prefix/suffix and a non-empty band.
        cap = max(0, (L - 2) // 3)
        lo = min(int(getattr(config, "mat_ffn_dyn_start", 0) or 0), cap)
        hi = L - min(int(getattr(config, "mat_ffn_dyn_end", 0) or 0), cap)
        if not (lo <= layer_idx < hi):
            # Fixed-compute prefix/suffix: full width for every token, unless
            # `mat_ffn_fixed_width` decouples them from intermediate_size.
            #
            # WHY THE KNOB EXISTS (2026-08-22). At matched ACTIVE params a
            # skewed tier prior pushes intermediate_size UP -- that is the whole
            # point, a bigger nest around the same expected cost -- but these
            # layers run the full d_ff on every token, so their share of the
            # active budget grows with the skew: 29% of the deep80 FFN budget at
            # uniform quarters, ~50% at d_ff=8513. Past that the fixed layers
            # eat the saving the router is buying. See docs/mat_ffn_nesting.md.
            fw = getattr(config, "mat_ffn_fixed_width", None)
            return SwiGLUMLP(
                hidden_size=config.hidden_size,
                intermediate_size=int(fw) if fw else config.intermediate_size,
                hidden_act=getattr(config, "hidden_act", "silu"),
                bias=getattr(config, "mlp_bias", False),
            )
        interval = max(1, int(getattr(config, "mat_ffn_route_interval", 1) or 1))
        # With mat_ffn_layer_kinds the kind filter above already picked the routed
        # layers (e.g. every "attn" slot of a 3:1 hybrid is every 4th layer), so the
        # index interval must not thin them a second time.
        if not kinds and (layer_idx - lo) % interval != 0:
            # ROUTE EVERY Nth LAYER. Measured: mat_ffn's per-layer routing
            # overhead is roughly constant, so total cost scales with the number
            # of routed layers -- 12 routed layers cost 0.4% over the backbone,
            # 70 cost 43%. Routing every layer is what made this the most
            # expensive of the new arms.
            #
            # The layers in between are NOT full width and NOT routed: they run
            # a plain SwiGLU at the BUDGET width for every token. That keeps
            # essentially all of the FLOP saving (the budget width IS the mean
            # routed width) on dense kernels with zero gather, zero sort and
            # zero sync, while the routed layers still supply the per-token
            # dynamism. Cheaper than full width AND cheaper than routing.
            widths = getattr(config, "mat_ffn_widths", None) or [
                config.intermediate_size // 4,
                config.intermediate_size // 2,
                config.intermediate_size,
            ]
            budget = float(getattr(config, "mat_ffn_budget", 0.0) or 0.0)
            if budget <= 0:
                budget = sum(widths) / len(widths)
            w = max(8, int(round(budget / 8.0)) * 8)  # keep the GEMM shape nice
            return SwiGLUMLP(
                hidden_size=config.hidden_size,
                intermediate_size=min(w, config.intermediate_size),
                hidden_act=getattr(config, "hidden_act", "silu"),
                bias=getattr(config, "mlp_bias", False),
            )
    if getattr(config, "use_mat_ffn", False):
        # W1 `mat_ffn`: per-TOKEN nested-width routing. Checked before
        # use_matformer because it is the routed superset of it.
        return RoutedMatFormerSwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            bias=getattr(config, "mlp_bias", False),
            granularities=getattr(config, "mat_ffn_widths", None),
            budget=getattr(config, "mat_ffn_budget", 0.0),
            aux_weight=getattr(config, "mat_ffn_aux_weight", 0.01),
            dispatch=getattr(config, "mat_ffn_dispatch", "sorted"),
            tier_fracs=getattr(config, "mat_ffn_tier_fracs", None),
            aux_mode=getattr(config, "mat_ffn_aux_mode", "mean"),
            explore_frac=float(getattr(config, "mat_ffn_explore_frac", 0.0) or 0.0),
            gate_mode=str(getattr(config, "mat_ffn_gate_mode", "ste") or "ste"),
            # NOT `or`-defaulted: 0.0 is falsy and would invert the meaning.
            mixture_frac=float(getattr(config, "mat_ffn_mixture_frac", 1.0)),
            route_mode=str(getattr(config, "mat_ffn_route_mode", "argmax") or "argmax"),
            route_thresh_lr=float(getattr(config, "mat_ffn_route_thresh_lr", 0.01)),
            # Absent from pre-2026-09-18 configs: those cells trained post_clip,
            # but the controller only acts in training, so eval is unaffected.
            route_ctrl=str(getattr(config, "mat_ffn_route_ctrl", "wish") or "wish"),
            cap_headroom=float(getattr(config, "mat_ffn_cap_headroom", 1.0) or 1.0),
            explore_from=str(getattr(config, "mat_ffn_explore_from", "target") or "target"),
            tier_scale_init=getattr(config, "mat_ffn_tier_scale_init", None),
            prior_bias_init=bool(getattr(config, "mat_ffn_prior_bias_init", False)),
            track_occupancy=bool(getattr(config, "mat_ffn_track_occupancy", True)),
            warmup_stages=int(getattr(config, "mat_ffn_warmup_stages", 4) or 4),
            warmup_top_weight=float(getattr(config, "mat_ffn_warmup_top_weight", 1.0) or 1.0),
        )
    if getattr(config, "use_matformer", False):
        return MatFormerSwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            bias=getattr(config, "mlp_bias", False),
            active_size=getattr(config, "matformer_active_size", None),
            granularities=getattr(config, "matformer_granularities", None),
        )
    return SwiGLUMLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=getattr(config, "hidden_act", "silu"),
        bias=getattr(config, "mlp_bias", False),
    )


from .mixture_of_depths import MoDGate, MoDOutput, MoDRouter, is_mod_layer


def _should_use_deltanet(config, layer_idx: int) -> bool:
    """Determine if this layer should use DeltaNet instead of attention.

    Args:
        config: Model configuration.
        layer_idx: Index of the layer (0-based).

    Returns:
        True if this layer should use DeltaNet.
    """
    if not getattr(config, "use_deltanet", False):
        return False

    # Explicit per-layer pattern takes priority
    pattern = getattr(config, "hybrid_attention_ratio", None)
    if pattern:
        return pattern[layer_idx % len(pattern)] == "delta"

    # Force first N layers to use standard attention
    first_n = getattr(config, "deltanet_first_n_layers", 0)
    if layer_idx < first_n:
        return False

    # Every N layers is attention (global refresh style)
    # e.g., every_n=4 means layers 3, 7, 11... use attention (0-indexed)
    every_n = getattr(config, "deltanet_every_n_layers", 0)
    if every_n <= 0:
        # every_n=0 means all DeltaNet
        return True

    # Use DeltaNet unless this is an "attention" layer
    # Layer uses attention if (layer_idx - first_n + 1) % every_n == 0
    adjusted_idx = layer_idx - first_n
    return (adjusted_idx + 1) % every_n != 0


def _should_use_griffin(config, layer_idx: int) -> bool:
    """Determine if this layer should use Griffin RG-LRU instead of attention.

    Args:
        config: Model configuration.
        layer_idx: Index of the layer (0-based).

    Returns:
        True if this layer should use RG-LRU.
    """
    if not getattr(config, "use_griffin", False):
        return False

    pattern = getattr(config, "griffin_pattern", None)
    if pattern:
        return pattern[layer_idx % len(pattern)] == "rglru"

    # Default: all layers are RG-LRU (pure Griffin)
    return True


def _ratio_entry_for_layer(config, layer_idx):
    """Return the hybrid_attention_ratio entry for a given layer (None if no ratio)."""
    if layer_idx is None:
        return None
    pattern = getattr(config, "hybrid_attention_ratio", None)
    if not pattern:
        return None
    return pattern[layer_idx % len(pattern)]


def _resolve_mla_class(attn_implementation):
    if attn_implementation == "flash_attention_2":
        # FlashAttention avoids the [B,H,S,S] score matrix the eager/SDPA MLA
        # paths materialise for the global layers (fatal at long context).
        return MLA_ATTENTION_CLASSES.get("flash_attention_2", UMichSLMMLAAttention)
    if attn_implementation == "sdpa":
        return MLA_ATTENTION_CLASSES.get("sdpa", UMichSLMMLAAttention)
    return MLA_ATTENTION_CLASSES.get("eager", UMichSLMMLAAttention)


def _resolve_deltanet_class(config):
    variant = getattr(config, "deltanet_variant", "gated")
    if variant == "rwkv7":
        # Lazy: rwkv7.py imports DeltaNetCache from deltanet.py.
        from .rwkv7 import RWKV7Mixer

        return RWKV7Mixer
    return DELTANET_CLASSES.get(variant, GatedDeltaNet)


def _resolve_swa_globals_class(attn_implementation, config=None):
    """Resolve the SWA-with-globals attention class. Lazy import to avoid cycles.

    With ``config.swa_use_registers`` the "swa" entries dispatch to the
    registers twin instead (pure-window self-attention + register-bank read;
    see ``swa_registers.py``) — the globals never enter the sequence.
    """
    # use_logit_sinks belongs here too: the per-head sink logit lives on
    # ScribeSinkMixin, so an arm with logit sinks but NO sink bank and NO scribe
    # bank (P2 `sink_diet`) would otherwise fall through to the plain registers
    # class and the knob would silently do nothing -- verified by a forward
    # A/B that came back bitwise identical before this line was fixed.
    if config is not None and (
        getattr(config, "swa_sink_count", 0) > 0
        or getattr(config, "use_scribe_bank", False)
        or getattr(config, "use_logit_sinks", False)
    ):
        from .swa_scribe import SWA_SCRIBE_CLASSES, SWAScribeAttention

        return SWA_SCRIBE_CLASSES.get(attn_implementation, SWAScribeAttention)
    if config is not None and getattr(config, "swa_use_registers", False):
        reg_count = getattr(config, "swa_register_count", None)
        total = int(getattr(config, "swa_global_count", 0))
        if reg_count is not None and 0 < int(reg_count) < total:
            # Hybrid split: a (smaller) prepended prefix remains, so the layer
            # must keep the full prefix machinery (flex path, num_protected_prefix)
            # AND add the register read.
            from .swa_registers import SWA_HYBRID_CLASSES, SWAHybridAttention

            return SWA_HYBRID_CLASSES.get(attn_implementation, SWAHybridAttention)
        from .swa_registers import SWA_REGISTERS_CLASSES, SWARegistersAttention

        return SWA_REGISTERS_CLASSES.get(attn_implementation, SWARegistersAttention)
    from .swa_with_globals import SWA_GLOBALS_CLASSES, SWAWithGlobalsAttention

    return SWA_GLOBALS_CLASSES.get(attn_implementation, SWAWithGlobalsAttention)


def _get_attention_class(config, layer_idx: int = None, _skip_hymba: bool = False):
    """Get the appropriate attention class based on config.

    Supports standard attention, MLA, DeltaNet (linear attention), Hymba, and the
    SWA + persistent globals layer used by attention_mode="swa_globals_mla".

    When ``hybrid_attention_ratio`` is set, the per-layer entry directly determines
    the dispatch:
      - "delta" -> DeltaNet (regardless of use_mla flag)
      - "mla"   -> MLA
      - "swa"   -> SWA + persistent globals
      - "attn"  -> standard attention
    This allows the 3:1 hybrid (e.g. ["delta","delta","delta","mla"]) to coexist
    with both ``use_deltanet=True`` and ``use_mla=True`` set on the config — a
    pattern that was previously rejected by config validation.

    Args:
        config: Model configuration.
        layer_idx: Layer index for per-layer configuration.
        _skip_hymba: If True, skip Hymba selection (used internally by
            HymbaHybridHead to select its inner attention class without recursion).

    Returns:
        The attention class to use.
    """
    # Hymba hybrid head (highest priority, wraps inner attention + SSM)
    if not _skip_hymba and getattr(config, "use_hymba", False):
        from .hymba import HymbaHybridHead

        return HymbaHybridHead

    # Check if this layer should use Griffin RG-LRU
    if layer_idx is not None and _should_use_griffin(config, layer_idx):
        return RGLRUAttention

    attn_implementation = getattr(config, "_attn_implementation", "eager")

    # When hybrid_attention_ratio is set, dispatch directly off the per-layer entry.
    # This is the path used by attention_mode in {"deltanet_mla", "swa_globals_mla"}
    # and by legacy hybrid configs that mix DeltaNet + standard attention.
    entry = _ratio_entry_for_layer(config, layer_idx)
    if entry is not None:
        if entry == "delta":
            return _resolve_deltanet_class(config)
        if entry == "mla":
            return _resolve_mla_class(attn_implementation)
        if entry == "swa":
            return _resolve_swa_globals_class(attn_implementation, config)
        if entry == "attn":
            if int(getattr(config, "attn_register_count", 0) or 0) > 0:
                from .swa_registers import FULL_REGISTERS_CLASSES, FullRegistersAttention

                return FULL_REGISTERS_CLASSES.get(attn_implementation, FullRegistersAttention)
            return ATTENTION_CLASSES.get(attn_implementation, UMichSLMAttention)
        # Fall through if entry is unrecognized — config validation should have caught it.

    # Legacy fallback: use_deltanet flag drives per-layer DeltaNet selection.
    if layer_idx is not None and _should_use_deltanet(config, layer_idx):
        return _resolve_deltanet_class(config)

    # Check for MLA
    use_mla = getattr(config, "use_mla", False)
    if use_mla:
        return _resolve_mla_class(attn_implementation)

    # Standard attention
    return ATTENTION_CLASSES.get(attn_implementation, UMichSLMAttention)


class UMichSLMDecoderLayer(nn.Module):
    """Single transformer decoder layer with pre-norm or post-norm architecture.

    Pre-norm (default):
        output = x + Attention(Norm(x))
        output = output + MLP(Norm(output))

    Post-norm:
        output = Norm(x + Attention(x))
        output = Norm(output + MLP(output))

    Supports MoDA (Mixture-of-Depths Attention) via optional depth KV cache.

    Args:
        config: Model configuration object.
        layer_idx: Index of this layer in the stack.

    Attributes:
        self_attn: Multi-head self-attention module.
        mlp: SwiGLU MLP module.
        input_layernorm: RMSNorm for attention sublayer.
        post_attention_layernorm: RMSNorm for MLP sublayer.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.norm_placement = getattr(config, "norm_placement", "pre")

        # Select attention implementation (supports both standard and MLA)
        attention_class = _get_attention_class(config, layer_idx)

        # Self-attention
        self.self_attn = attention_class(config=config, layer_idx=layer_idx)

        # MLP (SwiGLU, optionally MatFormer nested-FFN)
        self.mlp = _build_mlp(config, layer_idx)

        # Layer norms
        norm_class = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        self.input_layernorm = norm_class(
            config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
        )
        self.post_attention_layernorm = norm_class(
            config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
        )

        # MoDA FFN projection (optional)
        self.use_moda = getattr(config, "use_moda", False)
        if self.use_moda and getattr(config, "moda_use_ffn_kv", True):
            from .moda import MoDAFFNProjection

            num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
            head_dim = config.hidden_size // config.num_attention_heads
            self.moda_ffn_proj = MoDAFFNProjection(config.hidden_size, num_kv_heads, head_dim)
        else:
            self.moda_ffn_proj = None

        # State/prediction separation (modules/pause_stream.py). `pause_enabled`
        # is a PLAIN BOOL read in forward, not a config lookup, so the model can
        # phase separation on mid-run: the paper's cheap variants all start from
        # a standard checkpoint and switch over partway through the schedule.
        # Under torch.compile the switch costs one recompile, once per run.
        # Depth parameterization (configuration_umichslm.py, 2026-10-31): a residual
        # branch multiplier base/L (CompleteP, alpha = 1) and LayerNorm scaling
        # 1/sqrt(layer_idx + 1). Both are exactly 1.0 when off, and _branch/_ln_out
        # then return their input untouched, so default cells are bitwise unchanged.
        base = int(getattr(config, "depth_scale_base_layers", 0) or 0)
        self.residual_scale = base / config.num_hidden_layers if base > 0 else 1.0
        self.ln_scale = (
            1.0 / math.sqrt(layer_idx + 1) if getattr(config, "layernorm_scaling", False) else 1.0
        )
        if (self.residual_scale != 1.0 or self.ln_scale != 1.0) and self.norm_placement != "pre":
            raise NotImplementedError("depth_scale_base_layers / layernorm_scaling need pre-norm")
        self.pause_two_stream = bool(getattr(config, "use_pause_stream", False))
        self.pause_enabled = self.pause_two_stream
        self.pause_ffn = None
        if self.pause_two_stream:
            if self.norm_placement != "pre":
                raise NotImplementedError(
                    "use_pause_stream requires norm_placement='pre'; the post-norm "
                    "branch normalizes the residual after the add, which would mix "
                    "the two streams' statistics."
                )
            # The pause query reaches the mixer one of two ways: `precomputed_q`
            # (GQA and the registers/scribe SWA family: q_proj forked before
            # q_norm/RoPE) or `pause_query_states` (MLA and Gated DeltaNet: the
            # module derives Q from these rows itself, since MLA's Q is a
            # down/up/rope split and DeltaNet's runs through the short conv). A
            # module that accepts neither would silently attend with the STATE
            # stream's queries and the whole arm would be a no-op -- exactly the
            # class of silent-knob failure that has already invalidated two
            # ablations in this project. Fail at construction.
            import inspect

            params = inspect.signature(self.self_attn.forward).parameters
            if "pause_query_states" in params:
                self.pause_query_mode = "states"
            elif hasattr(self.self_attn, "q_proj") and "precomputed_q" in params:
                self.pause_query_mode = "q"
            else:
                raise NotImplementedError(
                    f"use_pause_stream needs {type(self.self_attn).__name__}.forward "
                    f"to accept precomputed_q (with a q_proj) or pause_query_states "
                    f"(layer {layer_idx}); without either the prediction stream would "
                    f"attend with the state stream's queries and the arm would "
                    f"measure nothing."
                )
            if getattr(config, "pause_shared_ffn", True):
                from .pause_stream import PauseSharedFFN

                self.pause_ffn = PauseSharedFFN(config.hidden_size)

    def _branch(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.residual_scale == 1.0 else x * self.residual_scale

    def _ln_out(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.ln_scale == 1.0 else x * self.ln_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        depth_kv_cache=None,
    ) -> Tuple[torch.FloatTensor, ...]:
        """Forward pass through the decoder layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
            attention_mask: Optional attention mask.
            position_ids: Optional position indices.
            past_key_value: Optional cached key/value states.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return updated cache.
            depth_kv_cache: Optional MoDADepthKVCache for depth attention.

        Returns:
            Tuple containing:
                - hidden_states: Output tensor of same shape as input
                - (optional) attention weights if output_attentions=True
                - (optional) present key/value cache if use_cache=True
        """
        residual = hidden_states

        if self.pause_two_stream and self.pause_enabled:
            # Two streams, concatenated along the batch dim with the STATE half
            # first (see modules/pause_stream.py). The only thing that differs
            # from an ordinary layer: K/V come from the state half for BOTH
            # halves, while Q comes from each half's own residual stream, so the
            # prediction stream reads the state stream's context and writes no
            # K/V of its own. One FA2 call at batch 2B, ordinary causal mask.
            normed = self._ln_out(self.input_layernorm(hidden_states))
            half = normed.shape[0] // 2
            state_normed = normed[:half]
            # Q source: each stream's own state. The precomputed_q fork is before
            # q_norm/RoPE, so the pause query still carries its own position's
            # rotation; pause_query_states hands the module the rows instead.
            if self.pause_query_mode == "q":
                q_kw = dict(precomputed_q=self.self_attn.q_proj(normed))
            else:
                q_kw = dict(pause_query_states=normed)
            attn_out, self_attn_weights, present_key_value = self.self_attn(
                # K/V source: the state half, repeated so both halves see it.
                hidden_states=torch.cat([state_normed, state_normed], dim=0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                depth_kv_cache=depth_kv_cache,
                **q_kw,
            )
            hidden_states = residual + self._branch(attn_out)

            residual = hidden_states
            normed_for_mlp = self._ln_out(self.post_attention_layernorm(hidden_states))
            if self.pause_ffn is not None:
                hidden_states = residual + self._branch(self.pause_ffn(self.mlp, normed_for_mlp))
            else:
                hidden_states = residual + self._branch(self.mlp(normed_for_mlp))
        elif self.norm_placement == "pre":
            # Pre-norm + Self-attention
            hidden_states = self._ln_out(self.input_layernorm(hidden_states))
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                depth_kv_cache=depth_kv_cache,
            )
            hidden_states = residual + self._branch(hidden_states)

            # Pre-norm + MLP
            residual = hidden_states
            normed_for_mlp = self._ln_out(self.post_attention_layernorm(hidden_states))

            # MoDA: project FFN input to depth KV
            if depth_kv_cache is not None and self.moda_ffn_proj is not None:
                k_ffn, v_ffn = self.moda_ffn_proj(normed_for_mlp)
                depth_kv_cache.append(self.layer_idx, k_ffn, v_ffn)

            hidden_states = self.mlp(normed_for_mlp)
            hidden_states = residual + self._branch(hidden_states)
        else:
            # Post-norm: Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                depth_kv_cache=depth_kv_cache,
            )
            hidden_states = self.input_layernorm(residual + hidden_states)

            # Post-norm: MLP
            residual = hidden_states

            # MoDA: project FFN input to depth KV
            if depth_kv_cache is not None and self.moda_ffn_proj is not None:
                k_ffn, v_ffn = self.moda_ffn_proj(hidden_states)
                depth_kv_cache.append(self.layer_idx, k_ffn, v_ffn)

            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_attention_layernorm(residual + hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class UMichSLMDecoderLayerWithParallelMLP(nn.Module):
    """Decoder layer with parallel attention and MLP computation.

    This variant computes attention and MLP in parallel (as in GPT-J, PaLM)
    which can improve throughput on some hardware:
        output = x + Attention(Norm(x)) + MLP(Norm(x))

    Args:
        config: Model configuration object.
        layer_idx: Index of this layer in the stack.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Select attention implementation (supports both standard and MLA)
        attention_class = _get_attention_class(config, layer_idx)

        # Self-attention
        self.self_attn = attention_class(config=config, layer_idx=layer_idx)

        # MLP (SwiGLU, optionally MatFormer nested-FFN)
        self.mlp = _build_mlp(config, layer_idx)

        # Single layer norm for parallel computation
        norm_class = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        self.input_layernorm = norm_class(
            config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
        )

        # MoDA FFN projection (optional)
        self.use_moda = getattr(config, "use_moda", False)
        if self.use_moda and getattr(config, "moda_use_ffn_kv", True):
            from .moda import MoDAFFNProjection

            num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
            head_dim = config.hidden_size // config.num_attention_heads
            self.moda_ffn_proj = MoDAFFNProjection(config.hidden_size, num_kv_heads, head_dim)
        else:
            self.moda_ffn_proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        depth_kv_cache=None,
    ) -> Tuple[torch.FloatTensor, ...]:
        """Forward pass with parallel attention and MLP.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
            attention_mask: Optional attention mask.
            position_ids: Optional position indices.
            past_key_value: Optional cached key/value states.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return updated cache.
            depth_kv_cache: Optional MoDADepthKVCache for depth attention.

        Returns:
            Tuple containing output and optional attention/cache.
        """
        residual = hidden_states

        # Apply normalization once
        hidden_states = self.input_layernorm(hidden_states)

        # MoDA: project FFN input to depth KV (same normalized input as MLP)
        if depth_kv_cache is not None and self.moda_ffn_proj is not None:
            k_ffn, v_ffn = self.moda_ffn_proj(hidden_states)
            depth_kv_cache.append(self.layer_idx, k_ffn, v_ffn)

        # Parallel attention and MLP
        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            depth_kv_cache=depth_kv_cache,
        )
        mlp_output = self.mlp(hidden_states)

        # Combine residual, attention, and MLP outputs
        hidden_states = residual + attn_output + mlp_output

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class UMichSLMMoEDecoderLayer(nn.Module):
    """Transformer decoder layer with Mixture of Experts MLP.

    This variant replaces the standard MLP with a sparse Mixture of Experts,
    where each token is routed to a subset of expert networks. This allows
    for significantly larger model capacity with minimal compute increase.

    The forward pass returns an additional MoEOutput containing auxiliary
    losses for load balancing.

    Args:
        config: Model configuration object with MoE settings.
        layer_idx: Index of this layer in the stack.
        num_experts: Number of experts for this layer. If None, uses config.num_experts.
            This allows progressive MoE where different layers have different expert counts.

    Attributes:
        self_attn: Multi-head self-attention module.
        moe: Mixture of Experts module (replaces standard MLP).
        input_layernorm: RMSNorm before attention.
        post_attention_layernorm: RMSNorm before MoE.
    """

    def __init__(self, config, layer_idx: int, num_experts: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.norm_placement = getattr(config, "norm_placement", "pre")

        # Use provided num_experts or fall back to config
        if num_experts is None:
            num_experts = config.num_experts
        self.num_experts = num_experts

        # Select attention implementation (supports both standard and MLA)
        attention_class = _get_attention_class(config, layer_idx)

        # Self-attention
        self.self_attn = attention_class(config=config, layer_idx=layer_idx)

        # MoE layer (replaces standard MLP)
        expert_intermediate_size = getattr(config, "moe_expert_intermediate_size", None)
        if expert_intermediate_size is None:
            expert_intermediate_size = config.intermediate_size

        num_experts_per_tok = min(config.num_experts_per_tok, num_experts)

        num_shared_experts = getattr(config, "num_shared_experts", 0)
        if num_shared_experts > 0:
            self.moe = SparseMoE(
                hidden_size=config.hidden_size,
                intermediate_size=expert_intermediate_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                num_shared_experts=num_shared_experts,
                hidden_act=getattr(config, "hidden_act", "silu"),
                mlp_bias=getattr(config, "mlp_bias", False),
                router_aux_loss_coef=getattr(config, "router_aux_loss_coef", 0.01),
                router_z_loss_coef=getattr(config, "router_z_loss_coef", 0.001),
            )
        else:
            self.moe = MoELayer(
                hidden_size=config.hidden_size,
                intermediate_size=expert_intermediate_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_act=getattr(config, "hidden_act", "silu"),
                mlp_bias=getattr(config, "mlp_bias", False),
                router_jitter=getattr(config, "router_jitter", 0.0),
                router_aux_loss_coef=getattr(config, "router_aux_loss_coef", 0.01),
                router_z_loss_coef=getattr(config, "router_z_loss_coef", 0.001),
            )

        # Layer norms
        norm_class = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        self.input_layernorm = norm_class(
            config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
        )
        self.post_attention_layernorm = norm_class(
            config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
        )

        # MoDA FFN projection (optional)
        self.use_moda = getattr(config, "use_moda", False)
        if self.use_moda and getattr(config, "moda_use_ffn_kv", True):
            from .moda import MoDAFFNProjection

            num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
            head_dim = config.hidden_size // config.num_attention_heads
            self.moda_ffn_proj = MoDAFFNProjection(config.hidden_size, num_kv_heads, head_dim)
        else:
            self.moda_ffn_proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        depth_kv_cache=None,
    ) -> Tuple[torch.FloatTensor, ...]:
        """Forward pass through the MoE decoder layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
            attention_mask: Optional attention mask.
            position_ids: Optional position indices.
            past_key_value: Optional cached key/value states.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return updated cache.
            output_router_logits: Whether to return router logits.
            depth_kv_cache: Optional MoDADepthKVCache for depth attention.

        Returns:
            Tuple containing:
                - hidden_states: Output tensor of same shape as input
                - (optional) attention weights if output_attentions=True
                - (optional) present key/value cache if use_cache=True
                - (optional) MoEOutput with router logits and aux losses
        """
        residual = hidden_states

        if self.norm_placement == "pre":
            # Pre-norm + Self-attention
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                depth_kv_cache=depth_kv_cache,
            )
            hidden_states = residual + hidden_states

            # Pre-norm + MoE
            residual = hidden_states
            normed_for_moe = self.post_attention_layernorm(hidden_states)

            # MoDA: project FFN input to depth KV
            if depth_kv_cache is not None and self.moda_ffn_proj is not None:
                k_ffn, v_ffn = self.moda_ffn_proj(normed_for_moe)
                depth_kv_cache.append(self.layer_idx, k_ffn, v_ffn)

            moe_output = self.moe(normed_for_moe)
            hidden_states = residual + moe_output.hidden_states
        else:
            # Post-norm: Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                depth_kv_cache=depth_kv_cache,
            )
            hidden_states = self.input_layernorm(residual + hidden_states)

            # Post-norm: MoE
            residual = hidden_states

            # MoDA: project FFN input to depth KV
            if depth_kv_cache is not None and self.moda_ffn_proj is not None:
                k_ffn, v_ffn = self.moda_ffn_proj(hidden_states)
                depth_kv_cache.append(self.layer_idx, k_ffn, v_ffn)

            moe_output = self.moe(hidden_states)
            hidden_states = self.post_attention_layernorm(residual + moe_output.hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (moe_output,)

        return outputs


class UMichSLMMoDDecoderLayer(nn.Module):
    """Transformer decoder layer with Mixture of Depths routing.

    This layer implements dynamic compute allocation where only a subset of
    tokens (determined by capacity) are processed through attention and MLP.
    Remaining tokens skip the layer via residual connection.

    The routing is learned: a linear projection produces a scalar weight for
    each token, and top-k tokens (based on capacity) are selected for processing.

    Args:
        config: Model configuration object.
        layer_idx: Index of this layer in the stack.

    Attributes:
        router: MoD router that decides which tokens to process.
        self_attn: Multi-head self-attention module.
        mlp: SwiGLU MLP module.
        gate: Gating mechanism for combining processed and skipped tokens.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.norm_placement = getattr(config, "norm_placement", "pre")

        # Determine MoD capacity for this layer. In stacking modes, blocks
        # marked is_inserted use config.mod_capacity_inserted (default 0.5);
        # original-and-grown blocks use config.mod_capacity (default 0.125).
        # The is_inserted flag is set by the staging package after model
        # construction (see staging.build_stage); we default to False here.
        self._is_inserted = False
        capacity_default = float(getattr(config, "mod_capacity", 0.125))

        # MoD router. The capacity is updated post-init by build_stage when
        # _is_inserted is set on this layer; we recompute below.
        self.router = MoDRouter(
            hidden_size=config.hidden_size,
            capacity=capacity_default,
            aux_loss_weight=getattr(config, "mod_aux_loss_weight", 0.01),
            use_aux_predictor=getattr(config, "mod_use_aux_predictor", False),
            gate_fn=getattr(config, "mod_gate_fn", "raw"),
            tanh_alpha=getattr(config, "mod_tanh_alpha", 0.2),
            capacity_warmup_steps=getattr(config, "mod_capacity_warmup_steps", 0),
        )

        # Number of leading positions (e.g. SWA persistent globals) that must
        # always be force-routed through this block. The model's forward stashes
        # the current value here at the start of every step.
        self.num_protected_prefix: int = 0

        # Cache values used by ``set_inserted`` to flip MoD capacity in-place
        # without re-instantiating the router.
        self._mod_capacity_default = capacity_default
        self._mod_capacity_inserted = float(getattr(config, "mod_capacity_inserted", 0.5))

        # MoD gate for combining outputs
        self.gate = MoDGate(
            config.hidden_size,
            symmetric_reweight=getattr(config, "mod_symmetric_reweight", False),
        )

        # Select attention implementation (supports both standard and MLA)
        attention_class = _get_attention_class(config, layer_idx)

        # Self-attention
        self.self_attn = attention_class(config=config, layer_idx=layer_idx)

        # MLP (SwiGLU, optionally MatFormer nested-FFN)
        self.mlp = _build_mlp(config, layer_idx)

        # Layer norms
        norm_class = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        self.input_layernorm = norm_class(
            config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
        )
        self.post_attention_layernorm = norm_class(
            config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
        )

        # MoDA FFN projection (optional)
        self.use_moda = getattr(config, "use_moda", False)
        if self.use_moda and getattr(config, "moda_use_ffn_kv", True):
            from .moda import MoDAFFNProjection

            num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
            head_dim = config.hidden_size // config.num_attention_heads
            self.moda_ffn_proj = MoDAFFNProjection(config.hidden_size, num_kv_heads, head_dim)
        else:
            self.moda_ffn_proj = None

    def set_inserted(self, is_inserted: bool) -> None:
        """Mark this MoD layer as 'inserted' (stage-time stack-grown block).

        When set, the router capacity is bumped from ``mod_capacity`` (default
        for original middle blocks) to ``mod_capacity_inserted`` (default 0.5,
        the value the stacking plan uses for inserted blocks). Original middle
        blocks and I/O blocks should remain dense — they should not be wrapped
        as MoD layers in the first place.
        """
        self._is_inserted = bool(is_inserted)
        self.router.capacity = (
            self._mod_capacity_inserted if self._is_inserted else self._mod_capacity_default
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        output_mod_info: Optional[bool] = False,
        depth_kv_cache=None,
    ) -> Tuple[torch.FloatTensor, ...]:
        """Forward pass through the MoD decoder layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
            attention_mask: Optional attention mask.
            position_ids: Optional position indices.
            past_key_value: Optional cached key/value states.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return updated cache.
            output_mod_info: Whether to return MoD routing info.
            depth_kv_cache: Optional MoDADepthKVCache for depth attention.

        Returns:
            Tuple containing:
                - hidden_states: Output tensor of same shape as input
                - (optional) attention weights if output_attentions=True
                - (optional) present key/value cache if use_cache=True
                - (optional) MoDOutput with routing info if output_mod_info=True
        """
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states

        # Get routing decisions. Pass the protected prefix (e.g. SWA globals)
        # so those positions always traverse this block; aux loss is computed
        # only over content tokens where routing is meaningful.
        router_weights, selected_mask, aux_loss = self.router(
            hidden_states,
            use_cache=use_cache,
            force_route_prefix=int(self.num_protected_prefix),
        )
        # Stash the selected mask so the instrumentation collector can compute
        # this block's routing fraction without a re-forward.
        self._last_selected_mask = selected_mask.detach()

        # Pre-norm + Self-attention
        normed_hidden = self.input_layernorm(hidden_states)
        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=normed_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            depth_kv_cache=depth_kv_cache,
        )

        # First residual connection (for attention)
        hidden_after_attn = residual + attn_output

        # Pre-norm + MLP
        normed_hidden = self.post_attention_layernorm(hidden_after_attn)

        # MoDA: project FFN input to depth KV
        if depth_kv_cache is not None and self.moda_ffn_proj is not None:
            k_ffn, v_ffn = self.moda_ffn_proj(normed_hidden)
            depth_kv_cache.append(self.layer_idx, k_ffn, v_ffn)

        mlp_output = self.mlp(normed_hidden)

        # Combined block output (attention + MLP contribution)
        block_output = attn_output + mlp_output

        # Apply MoD gating
        hidden_states = self.gate(
            hidden_states=residual,
            block_output=block_output,
            router_weights=router_weights,
            selected_mask=selected_mask,
        )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_mod_info:
            mod_output = MoDOutput(
                hidden_states=hidden_states,
                router_weights=router_weights,
                selected_mask=selected_mask,
                aux_loss=aux_loss,
            )
            outputs += (mod_output,)

        return outputs


class UMichSLMMoDMLPDecoderLayer(nn.Module):
    """MoD-MLP decoder layer: routing gates ONLY the FFN sublayer (Arm A).

    Attention (and therefore all KV / recurrent cross-token state) runs dense
    for every token; the router decides per token whether the FFN contribution
    is added. Structural consequences vs full-block MoD:

    - no KV-cache divergence at decode time (attention state is never routed),
    - safe alongside recurrent token mixers (the FFN is position-wise; F3),
    - total bypass is impossible (every token gets full attention every layer).

    Two routing sources:
    - *own router* (per-layer variant): this layer owns a ``MoDRouter`` applied
      to the post-attention normed hidden state.
    - *external routing* (block-atomic MLP variant, ``own_router=False``): the
      model computes one decision at the block head and passes
      ``mod_routing=(router_weights, selected_mask)`` into every member layer;
      the aux loss is emitted once at model level.

    When routing is uniform across rows (top-k with a forced prefix guarantees
    this) and no cache is in use, the FFN is executed only on the selected
    tokens (gather -> MLP -> scatter) — the FFN is position-wise, so unlike the
    attention gather path this realizes the FLOP savings exactly with no mask
    surgery. Ragged or cached decisions fall back to the dense-compute
    ``MoDGate`` path with identical math.

    Args:
        config: Model configuration object.
        layer_idx: Index of this layer in the stack.
        own_router: Whether this layer owns its router (per-layer variant).
    """

    def __init__(self, config, layer_idx: int, own_router: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.mod_gather_bucket = int(getattr(config, "mod_gather_bucket", 128))
        if getattr(config, "norm_placement", "pre") != "pre":
            raise ValueError(
                "UMichSLMMoDMLPDecoderLayer supports norm_placement='pre' only "
                f"(got {config.norm_placement!r})."
            )

        if own_router:
            self.router = MoDRouter(
                hidden_size=config.hidden_size,
                capacity=float(getattr(config, "mod_capacity", 0.125)),
                aux_loss_weight=getattr(config, "mod_aux_loss_weight", 0.01),
                use_aux_predictor=getattr(config, "mod_use_aux_predictor", False),
                gate_fn=getattr(config, "mod_gate_fn", "raw"),
                tanh_alpha=getattr(config, "mod_tanh_alpha", 0.2),
                capacity_warmup_steps=getattr(config, "mod_capacity_warmup_steps", 0),
                router_mode=getattr(config, "mod_router_mode", "topk"),
                route_bias_lr=getattr(config, "mod_route_bias_lr", 1e-3),
            )
        else:
            self.router = None

        self.gate = MoDGate(
            config.hidden_size,
            symmetric_reweight=getattr(config, "mod_symmetric_reweight", False),
        )

        # Set by the model's forward each step (SWA persistent globals).
        self.num_protected_prefix: int = 0

        self.self_attn = _get_attention_class(config, layer_idx)(config=config, layer_idx=layer_idx)
        self.mlp = _build_mlp(config, layer_idx)

        norm_class = get_rmsnorm_class(
            zero_centered=getattr(config, "use_zero_centered_rmsnorm", False)
        )
        self.input_layernorm = norm_class(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self.post_attention_layernorm = norm_class(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )

        # MoDA FFN projection (optional; parity with the other layer classes)
        self.use_moda = getattr(config, "use_moda", False)
        if self.use_moda and getattr(config, "moda_use_ffn_kv", True):
            from .moda import MoDAFFNProjection

            num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
            head_dim = config.hidden_size // config.num_attention_heads
            self.moda_ffn_proj = MoDAFFNProjection(config.hidden_size, num_kv_heads, head_dim)
        else:
            self.moda_ffn_proj = None

    def _gated_mlp(
        self,
        hidden_after_attn: torch.Tensor,
        normed_for_mlp: torch.Tensor,
        router_weights: torch.Tensor,
        selected_mask: torch.Tensor,
        use_cache: bool,
    ) -> torch.Tensor:
        """Apply the FFN only where routed; declined tokens pass through bitwise."""
        from .mixture_of_depths import mod_gather_index

        # Compact path: run the FFN on only the selected tokens. Valid because
        # the FFN is position-wise; requires uniform per-row counts and no
        # symmetric reweighting (STRing perturbs skipped tokens too).
        if not use_cache and not self.gate.symmetric_reweight:
            gathered_idx = mod_gather_index(selected_mask, bucket=self.mod_gather_bucket)
            if gathered_idx is not None:
                gather_idx, gather_valid = gathered_idx
                B, T, H = normed_for_mlp.shape
                idx_h = gather_idx.unsqueeze(-1).expand(B, gather_idx.shape[1], H)
                compact = torch.gather(normed_for_mlp, 1, idx_h)
                mlp_out = self.mlp(compact)
                w = torch.gather(router_weights, 1, gather_idx).unsqueeze(-1)
                if gather_valid is not None:
                    # Ragged (threshold-router) gather: a padding slot indexes a
                    # real but UNSELECTED position. The FFN is position-wise so
                    # there is no attention contamination, but the scatter would
                    # still write `hidden + w*mlp` over a token that was declined.
                    # Zeroing its weight makes the write a bitwise pass-through.
                    w = w * gather_valid.unsqueeze(-1).to(w.dtype)
                gathered = torch.gather(hidden_after_attn, 1, idx_h) + w * mlp_out
                out = hidden_after_attn.clone()
                out.scatter_(1, idx_h, gathered)
                return out

        # Dense fallback (ragged selection / cached decode / STRing): compute
        # the FFN everywhere and gate — identical math for selected tokens,
        # bitwise pass-through for declined ones.
        mlp_output = self.mlp(normed_for_mlp)
        return self.gate(
            hidden_states=hidden_after_attn,
            block_output=mlp_output,
            router_weights=router_weights,
            selected_mask=selected_mask,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        output_mod_info: Optional[bool] = False,
        mod_routing: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        depth_kv_cache=None,
    ) -> Tuple[torch.FloatTensor, ...]:
        """Forward pass: dense attention, routed FFN.

        Args:
            mod_routing: Optional externally-computed ``(router_weights,
                selected_mask)`` from a block-head router (block-atomic MLP
                variant). Required when the layer was built with
                ``own_router=False``.

        Returns:
            Tuple of hidden_states plus the usual optional entries
            (attention weights, present KV, MoDOutput).
        """
        residual = hidden_states

        # Dense attention for every token — the defining invariant of Arm A.
        normed = self.input_layernorm(hidden_states)
        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            depth_kv_cache=depth_kv_cache,
        )
        hidden_after_attn = residual + attn_output

        normed_for_mlp = self.post_attention_layernorm(hidden_after_attn)

        # MoDA: project FFN input to depth KV (dense — routing never gates it)
        if depth_kv_cache is not None and self.moda_ffn_proj is not None:
            k_ffn, v_ffn = self.moda_ffn_proj(normed_for_mlp)
            depth_kv_cache.append(self.layer_idx, k_ffn, v_ffn)

        aux_loss = None
        if mod_routing is not None:
            router_weights, selected_mask = mod_routing
        else:
            if self.router is None:
                raise RuntimeError(
                    "MoDMLP layer built with own_router=False requires mod_routing "
                    "from the block-head router."
                )
            # Route on the post-attention normed hidden state.
            router_weights, selected_mask, aux_loss = self.router(
                normed_for_mlp,
                use_cache=use_cache,
                force_route_prefix=int(self.num_protected_prefix),
            )
        # For the instrumentation collector (routing-fraction metrics).
        self._last_selected_mask = selected_mask.detach()

        hidden_states = self._gated_mlp(
            hidden_after_attn, normed_for_mlp, router_weights, selected_mask, bool(use_cache)
        )

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_mod_info:
            outputs += (
                MoDOutput(
                    hidden_states=hidden_states,
                    router_weights=router_weights,
                    selected_mask=selected_mask,
                    aux_loss=aux_loss,
                ),
            )
        return outputs


def get_decoder_layer_class(
    parallel_mlp: bool = False,
    use_moe: bool = False,
    use_mod: bool = False,
    use_mod_mlp: bool = False,
):
    """Get the decoder layer class.

    Args:
        parallel_mlp: Whether to use parallel MLP architecture.
        use_moe: Whether to use Mixture of Experts.
        use_mod: Whether to use Mixture of Depths.
        use_mod_mlp: Whether to use MoD-MLP (FFN-only routing; wins over use_mod).

    Returns:
        The decoder layer class.
    """
    if use_moe:
        return UMichSLMMoEDecoderLayer
    if use_mod_mlp:
        return UMichSLMMoDMLPDecoderLayer
    if use_mod:
        return UMichSLMMoDDecoderLayer
    if parallel_mlp:
        return UMichSLMDecoderLayerWithParallelMLP
    return UMichSLMDecoderLayer
