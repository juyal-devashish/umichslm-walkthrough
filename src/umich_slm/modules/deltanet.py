"""Gated DeltaNet and Kimi Delta Attention implementations.

This module implements linear attention alternatives to standard softmax attention,
combining Mamba2's data-dependent gating with the delta rule memory update.

Key concepts:
- O(n) complexity instead of O(n²) quadratic attention
- Fixed-size recurrent state instead of growing KV cache
- Hybrid architectures supported (mixing with standard attention)

Two variants are implemented:
1. GatedDeltaNet (Qwen3-Next style): Scalar gating per head
2. KimiDeltaAttention (Kimi-K2 style): Channel-wise gating for finer control

References:
- Paper: https://arxiv.org/abs/2412.06464 "Gated Delta Networks"
- FLA Library: https://github.com/fla-org/flash-linear-attention
- NVlabs: https://github.com/NVlabs/GatedDeltaNet
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput

logger = logging.getLogger(__name__)

# Try to import FLA library (optimized Triton kernels)
FLA_AVAILABLE = False
try:
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    FLA_AVAILABLE = True
except ImportError:
    pass

# Track whether we've warned about FLA being unavailable
_FLA_WARNING_ISSUED = False


def _warn_fla_unavailable():
    """Issue a one-time warning that FLA library is not available."""
    global _FLA_WARNING_ISSUED
    if not _FLA_WARNING_ISSUED and not FLA_AVAILABLE:
        logger.warning(
            "flash-linear-attention (FLA) library not found. DeltaNet will use a naive "
            "PyTorch implementation which is significantly slower for long sequences. "
            "For better performance on CUDA, install FLA: pip install flash-linear-attention"
        )
        _FLA_WARNING_ISSUED = True


@dataclass
class DeltaNetCache:
    """Cache for DeltaNet layers during incremental generation.

    DeltaNet requires caching two types of state:
    1. recurrent_state: The delta rule memory [batch, heads, key_dim, value_dim]
    2. conv_state: Last (kernel_size - 1) hidden states for short convolution

    Attributes:
        recurrent_state: The delta rule recurrent memory matrix.
        conv_state: Cached hidden states for short convolution, shape
            [batch, hidden_size, kernel_size - 1]. None if conv is disabled.
        seen_tokens: Total number of tokens represented by this cache. This is
            optional metadata used to round-trip through legacy cache formats.
    """

    recurrent_state: Optional[torch.FloatTensor] = None
    conv_state: Optional[torch.FloatTensor] = None
    seen_tokens: Optional[int] = None


@dataclass
class DeltaNetOutput(ModelOutput):
    """Output type for DeltaNet layers.

    Attributes:
        hidden_states: Output hidden states.
        recurrent_state: The recurrent state for caching during generation.
    """

    hidden_states: torch.FloatTensor = None
    recurrent_state: Optional[torch.FloatTensor] = None


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet attention (Qwen3-Next variant).

    Implements linear attention with scalar gating per head and delta rule
    memory updates. This provides O(n) complexity with fixed-size recurrent
    state instead of O(n²) attention with growing KV cache.

    The delta rule update is:
        S_t = gate * S_{t-1} + beta * (k_t ⊗ v_t)
        o_t = q_t @ S_t

    Args:
        config: Model configuration object.
        layer_idx: Index of this layer in the stack.

    Attributes:
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        gate_proj: Gate projection (scalar per head, or channel-wise for KDA).
        beta_proj: Beta projection (learning rate for delta rule).
        o_proj: Output projection.
    """

    # Subclasses can override this to change gating behavior
    _use_channel_wise_gating: bool = False

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()

        # Warn once if FLA is not available
        _warn_fla_unavailable()

        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = getattr(config, "deltanet_num_heads", None) or config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.expand_k = getattr(config, "deltanet_expand_k", 1)
        self.expand_v = getattr(config, "deltanet_expand_v", 1)

        # Short convolution settings
        self.use_short_conv = getattr(config, "deltanet_use_short_conv", True)
        self.conv_kernel_size = getattr(config, "deltanet_conv_kernel_size", 4)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size} and num_heads={self.num_heads})"
            )

        # Projections
        # Q and K must have same dimension for delta rule: out = q @ state where state is [D_k, D_v]
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim * self.expand_k,  # Match K dimension
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim * self.expand_k,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim * self.expand_v,
            bias=False,
        )

        # Gating projections - dimension depends on gating type
        gate_dim = self._get_gate_dim()
        self.gate_proj = nn.Linear(self.hidden_size, gate_dim, bias=False)
        self.beta_proj = nn.Linear(self.hidden_size, gate_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim * self.expand_v,
            self.hidden_size,
            bias=False,
        )

        # Optional short convolution for local context (like in Mamba)
        if self.use_short_conv:
            self.conv = nn.Conv1d(
                self.hidden_size,
                self.hidden_size,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_kernel_size - 1,
                groups=self.hidden_size,
            )
        else:
            self.conv = None

    def _get_gate_dim(self) -> int:
        """Get the output dimension for gate projections.

        Returns:
            num_heads for scalar gating, num_heads * head_dim * expand_v for channel-wise.
        """
        if self._use_channel_wise_gating:
            return self.num_heads * self.head_dim * self.expand_v
        return self.num_heads

    def _apply_short_conv(
        self,
        hidden_states: torch.Tensor,
        conv_state: Optional[torch.Tensor],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply short convolution with state caching for incremental decoding.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            conv_state: Cached conv state from previous step, or None.
            use_cache: Whether to compute and return new conv state.

        Returns:
            Tuple of (output hidden states, new conv state or None).
        """
        if self.conv is None:
            return hidden_states, None

        batch_size, seq_len, _ = hidden_states.shape

        # Conv1d expects (batch, channels, seq_len)
        conv_input = hidden_states.transpose(1, 2)  # [B, H, L]

        # For incremental decoding, prepend cached conv state
        if conv_state is not None:
            conv_input = torch.cat([conv_state, conv_input], dim=2)

        # Apply convolution
        conv_out = self.conv(conv_input)

        # Extract output corresponding to current input positions
        if conv_state is not None:
            conv_out = conv_out[
                :, :, self.conv_kernel_size - 1 : self.conv_kernel_size - 1 + seq_len
            ]
        else:
            conv_out = conv_out[:, :, :seq_len]

        output = hidden_states + conv_out.transpose(1, 2)

        # Compute new conv state if caching
        new_conv_state = None
        if use_cache:
            if conv_state is not None:
                new_conv_state = conv_input[:, :, -(self.conv_kernel_size - 1) :].contiguous()
            else:
                cache_len = min(seq_len, self.conv_kernel_size - 1)
                if cache_len < self.conv_kernel_size - 1:
                    padding = torch.zeros(
                        batch_size,
                        self.hidden_size,
                        self.conv_kernel_size - 1 - cache_len,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                    new_conv_state = torch.cat([padding, conv_input[:, :, -cache_len:]], dim=2)
                else:
                    new_conv_state = conv_input[:, :, -(self.conv_kernel_size - 1) :]
            new_conv_state = new_conv_state.contiguous()

        return output, new_conv_state

    def _compute_gates(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gate and beta values from hidden states.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Tuple of (gate, beta) tensors. Shape depends on gating type:
            - Scalar gating: [batch, seq_len, num_heads]
            - Channel-wise: [batch, seq_len, num_heads, value_dim]
        """
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        beta = torch.sigmoid(self.beta_proj(hidden_states))

        if self._use_channel_wise_gating:
            v_dim = self.head_dim * self.expand_v
            gate = gate.view(batch_size, seq_len, self.num_heads, v_dim)
            beta = beta.view(batch_size, seq_len, self.num_heads, v_dim)

        return gate, beta

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[DeltaNetCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        depth_kv_cache=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[DeltaNetCache]]:
        """Forward pass with HuggingFace-compatible signature.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
            attention_mask: Not used by DeltaNet but kept for interface compatibility.
            position_ids: Not used by DeltaNet (no position encoding needed).
            past_key_value: DeltaNetCache from previous step for incremental decoding.
            output_attentions: Not supported for linear attention, always returns None.
            use_cache: Whether to return the cache for incremental generation.

        Returns:
            Tuple of:
                - Output tensor of shape (batch, seq_len, hidden_size)
                - None (attention weights not available for linear attention)
                - DeltaNetCache if use_cache=True, else None
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Extract cached states
        recurrent_state = None
        conv_state = None
        if past_key_value is not None:
            recurrent_state = past_key_value.recurrent_state
            conv_state = past_key_value.conv_state

        # Apply short convolution
        hidden_states, new_conv_state = self._apply_short_conv(hidden_states, conv_state, use_cache)

        # Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to heads: [batch, seq_len, num_heads, dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim * self.expand_k)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim * self.expand_k)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim * self.expand_v)

        # Compute gates
        gate, beta = self._compute_gates(hidden_states, batch_size, seq_len)

        # Choose implementation
        # FLA only supports scalar gating and doesn't have backward pass for fused_recurrent_gated_delta_rule
        # Use naive implementation when:
        # 1. FLA not available
        # 2. Not on CUDA
        # 3. Using channel-wise gating (KDA)
        # 4. Training (need backward pass) - FLA's fused_recurrent_gated_delta_rule has no backward
        use_fla = (
            FLA_AVAILABLE
            and hidden_states.is_cuda
            and not self._use_channel_wise_gating
            and not self.training  # FLA has no backward pass for gated delta rule
        )
        if use_fla:
            output, new_recurrent_state = self._fla_forward(q, k, v, gate, beta, recurrent_state)
        else:
            output, new_recurrent_state = self._naive_forward(q, k, v, gate, beta, recurrent_state)

        # Output projection
        output = output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)

        # Build cache
        present = None
        if use_cache:
            present = DeltaNetCache(
                recurrent_state=new_recurrent_state,
                conv_state=new_conv_state,
            )

        return output, None, present

    def _naive_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Naive PyTorch implementation for reference/CPU fallback.

        This is a sequential implementation that processes one timestep at a time.
        It's slower than the FLA Triton kernels but works on any device.
        Handles both scalar gating (GatedDeltaNet) and channel-wise gating (KDA).

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim].
            k: Key tensor [batch, seq_len, num_heads, key_dim].
            v: Value tensor [batch, seq_len, num_heads, value_dim].
            gate: Gate tensor. Shape is [batch, seq_len, num_heads] for scalar gating,
                or [batch, seq_len, num_heads, value_dim] for channel-wise.
            beta: Beta tensor, same shape as gate.
            state: Previous recurrent state [batch, num_heads, key_dim, value_dim].

        Returns:
            Tuple of (output, new_state).
        """
        B, L, H, D_k = k.shape
        D_v = v.shape[-1]

        # Initialize state if needed
        if state is None:
            state = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=q.dtype)

        outputs = []
        for t in range(L):
            # Get current timestep
            k_t = k[:, t]  # [B, H, D_k]
            v_t = v[:, t]  # [B, H, D_v]
            q_t = q[:, t]  # [B, H, D_q]

            # Outer product k^T @ v -> [B, H, D_k, D_v]
            kv = torch.einsum("bhk,bhv->bhkv", k_t, v_t)

            # Get gates for this timestep and expand for broadcasting
            if self._use_channel_wise_gating:
                # Channel-wise: gate is [B, L, H, D_v]
                g_t = gate[:, t].unsqueeze(2)  # [B, H, 1, D_v]
                b_t = beta[:, t].unsqueeze(2)  # [B, H, 1, D_v]
            else:
                # Scalar: gate is [B, L, H]
                g_t = gate[:, t, :, None, None]  # [B, H, 1, 1]
                b_t = beta[:, t, :, None, None]  # [B, H, 1, 1]

            # Delta rule update: S_t = gate * S_{t-1} + beta * (k ⊗ v)
            state = g_t * state + b_t * kv

            # Query the state: o_t = q_t @ S_t
            out_t = torch.einsum("bhd,bhdv->bhv", q_t, state)
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)  # [B, L, H, D_v]
        return output, state

    def _fla_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FLA library forward with optimized Triton kernels.

        Only supports scalar gating (GatedDeltaNet). For channel-wise gating,
        use _naive_forward instead.

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim].
            k: Key tensor [batch, seq_len, num_heads, key_dim].
            v: Value tensor [batch, seq_len, num_heads, value_dim].
            gate: Gate tensor [batch, seq_len, num_heads].
            beta: Beta tensor [batch, seq_len, num_heads].
            state: Previous recurrent state.

        Returns:
            Tuple of (output, new_state).
        """
        # Transpose for FLA: [B, H, L, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        gate = gate.transpose(1, 2).unsqueeze(-1).contiguous()  # [B, H, L, 1]
        beta = beta.transpose(1, 2).unsqueeze(-1).contiguous()  # [B, H, L, 1]

        output, new_state = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            gate,
            beta,
            initial_state=state,
            output_final_state=True,
        )

        # Transpose back: [B, L, H, D]
        return output.transpose(1, 2), new_state


class KimiDeltaAttention(GatedDeltaNet):
    """Kimi Delta Attention (KDA) - channel-wise gating variant.

    Extends GatedDeltaNet with per-channel gating for finer-grained control.
    Instead of a scalar gate per head, KDA uses a separate gate value for
    each channel in the value dimension.

    This provides more expressiveness at the cost of additional parameters
    and computation. Used in Kimi-K2 (1T parameter MoE model).

    Note: FLA Triton kernels don't support channel-wise gating, so KDA
    always uses the naive PyTorch implementation.

    Args:
        config: Model configuration object.
        layer_idx: Index of this layer in the stack.
    """

    _use_channel_wise_gating: bool = True


# Registry for DeltaNet variants
DELTANET_CLASSES = {
    "gated": GatedDeltaNet,
    "kda": KimiDeltaAttention,
}


def get_deltanet_class(variant: str = "gated"):
    """Get the DeltaNet class for the given variant.

    Args:
        variant: One of "gated" (Qwen3-Next style) or "kda" (Kimi-K2 style).

    Returns:
        The DeltaNet class.
    """
    if variant not in DELTANET_CLASSES:
        raise ValueError(
            f"Unknown DeltaNet variant: {variant}. " f"Available: {list(DELTANET_CLASSES.keys())}"
        )
    return DELTANET_CLASSES[variant]
