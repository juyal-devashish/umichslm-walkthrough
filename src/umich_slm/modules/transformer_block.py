"""Transformer decoder layer implementation.

This module implements a complete transformer decoder layer that combines
attention, MLP, and normalization components in a pre-norm architecture.
Includes support for Mixture of Experts (MoE) layers.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .attention import UMichSLMAttention, ATTENTION_CLASSES
from .mla import MLA_ATTENTION_CLASSES, UMichSLMMLAAttention
from .deltanet import DELTANET_CLASSES, GatedDeltaNet
from .griffin import GRIFFIN_CLASSES, RGLRUAttention
from .rmsnorm import get_rmsnorm_class
from .swiglu import SwiGLUMLP
from .moe import MoELayer, SparseMoE, MoEOutput
from .mixture_of_depths import MoDRouter, MoDGate, MoDOutput, is_mod_layer


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
    if attn_implementation == "sdpa":
        return MLA_ATTENTION_CLASSES.get("sdpa", UMichSLMMLAAttention)
    return MLA_ATTENTION_CLASSES.get("eager", UMichSLMMLAAttention)


def _resolve_deltanet_class(config):
    variant = getattr(config, "deltanet_variant", "gated")
    return DELTANET_CLASSES.get(variant, GatedDeltaNet)


def _resolve_swa_globals_class(attn_implementation):
    """Resolve the SWA-with-globals attention class. Lazy import to avoid cycles."""
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
            return _resolve_swa_globals_class(attn_implementation)
        if entry == "attn":
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

        # MLP (SwiGLU)
        self.mlp = SwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            bias=getattr(config, "mlp_bias", False),
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

            # Pre-norm + MLP
            residual = hidden_states
            normed_for_mlp = self.post_attention_layernorm(hidden_states)

            # MoDA: project FFN input to depth KV
            if depth_kv_cache is not None and self.moda_ffn_proj is not None:
                k_ffn, v_ffn = self.moda_ffn_proj(normed_for_mlp)
                depth_kv_cache.append(self.layer_idx, k_ffn, v_ffn)

            hidden_states = self.mlp(normed_for_mlp)
            hidden_states = residual + hidden_states
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

        # MLP (SwiGLU)
        self.mlp = SwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            bias=getattr(config, "mlp_bias", False),
        )

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
        )

        # Number of leading positions (e.g. SWA persistent globals) that must
        # always be force-routed through this block. The model's forward stashes
        # the current value here at the start of every step.
        self.num_protected_prefix: int = 0

        # Cache values used by ``set_inserted`` to flip MoD capacity in-place
        # without re-instantiating the router.
        self._mod_capacity_default = capacity_default
        self._mod_capacity_inserted = float(
            getattr(config, "mod_capacity_inserted", 0.5)
        )

        # MoD gate for combining outputs
        self.gate = MoDGate(config.hidden_size)

        # Select attention implementation (supports both standard and MLA)
        attention_class = _get_attention_class(config, layer_idx)

        # Self-attention
        self.self_attn = attention_class(config=config, layer_idx=layer_idx)

        # MLP (SwiGLU)
        self.mlp = SwiGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            bias=getattr(config, "mlp_bias", False),
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


def get_decoder_layer_class(
    parallel_mlp: bool = False,
    use_moe: bool = False,
    use_mod: bool = False,
):
    """Get the decoder layer class.

    Args:
        parallel_mlp: Whether to use parallel MLP architecture.
        use_moe: Whether to use Mixture of Experts.
        use_mod: Whether to use Mixture of Depths.

    Returns:
        The decoder layer class.
    """
    if use_moe:
        return UMichSLMMoEDecoderLayer
    if use_mod:
        return UMichSLMMoDDecoderLayer
    if parallel_mlp:
        return UMichSLMDecoderLayerWithParallelMLP
    return UMichSLMDecoderLayer
