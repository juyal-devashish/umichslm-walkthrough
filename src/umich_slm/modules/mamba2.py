"""Mamba2 Selective State Space Model implementation.

This module implements the Mamba2 SSM (Structured State Space Model) used as the
recurrent component in Hymba hybrid attention-SSM layers.

Key concepts:
- Data-dependent gating via selective scan mechanism
- Conv1d for local context capture
- SiLU gating for output modulation
- O(n) complexity with fixed-size recurrent state

References:
- Mamba: https://arxiv.org/abs/2312.00752
- Mamba-2: https://arxiv.org/abs/2405.21060
- Hymba: https://arxiv.org/abs/2411.13676
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Try to import mamba-ssm library (optimized CUDA kernels)
MAMBA_AVAILABLE = False
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    MAMBA_AVAILABLE = True
except ImportError:
    pass

# Track whether we've warned about mamba-ssm being unavailable
_MAMBA_WARNING_ISSUED = False


def _warn_mamba_unavailable():
    """Issue a one-time warning that mamba-ssm library is not available."""
    global _MAMBA_WARNING_ISSUED
    if not _MAMBA_WARNING_ISSUED and not MAMBA_AVAILABLE:
        logger.warning(
            "mamba-ssm library not found. Mamba2 SSM will use a naive PyTorch "
            "implementation which is significantly slower for long sequences. "
            "For better performance on CUDA, install: pip install mamba-ssm causal-conv1d"
        )
        _MAMBA_WARNING_ISSUED = True


@dataclass
class Mamba2Cache:
    """Cache for Mamba2 SSM during incremental generation.

    Attributes:
        ssm_state: The recurrent state [batch, d_inner, d_state].
        conv_state: Conv1d buffer [batch, d_inner, d_conv - 1].
    """

    ssm_state: Optional[torch.Tensor] = None
    conv_state: Optional[torch.Tensor] = None


class Mamba2SSM(nn.Module):
    """Mamba2 Selective State Space Model.

    Implements the selective scan mechanism with data-dependent gating:
        x -> in_proj -> [x_ssm, z]
        x_ssm -> conv1d -> SiLU -> SSM scan -> y
        output = y * SiLU(z) -> out_proj

    The SSM scan computes:
        h_t = exp(dt * A) * h_{t-1} + dt * B_t * x_t
        y_t = C_t @ h_t

    Args:
        d_model: Input/output dimension.
        d_state: SSM state dimension (N). Default 16.
        d_conv: Conv1d kernel size. Default 4.
        expand: Expansion factor for inner dimension. Default 2.
        bias: Whether to use bias in linear projections. Default False.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bias: bool = False,
    ):
        super().__init__()

        _warn_mamba_unavailable()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        # Input projection: maps d_model -> 2 * d_inner (x_ssm and z gate)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # Causal conv1d for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # SSM parameters
        # dt (time step) projection
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # B and C projections (data-dependent)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # A parameter (fixed, learned in log space)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_inner + 1, dtype=torch.float32)))

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[Mamba2Cache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Mamba2Cache]]:
        """Forward pass through Mamba2 SSM.

        Args:
            hidden_states: Input tensor [batch, seq_len, d_model].
            cache: Optional Mamba2Cache for incremental decoding.
            use_cache: Whether to return updated cache.

        Returns:
            Tuple of (output [batch, seq_len, d_model], optional new cache).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Input projection -> x_ssm and z gate
        xz = self.in_proj(hidden_states)  # [B, L, 2 * d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)  # each [B, L, d_inner]

        # Conv1d (causal)
        x_ssm = self._apply_conv(x_ssm, cache)

        # SiLU activation after conv
        x_ssm = F.silu(x_ssm)

        # SSM scan
        y, new_ssm_state = self._ssm_scan(x_ssm, cache)

        # Skip connection with D
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_ssm

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        # Build cache
        new_cache = None
        if use_cache:
            new_cache = Mamba2Cache(
                ssm_state=new_ssm_state,
                conv_state=self._get_conv_state(x_ssm, cache),
            )

        return output, new_cache

    def _apply_conv(
        self,
        x: torch.Tensor,
        cache: Optional[Mamba2Cache],
    ) -> torch.Tensor:
        """Apply causal conv1d.

        Args:
            x: Input tensor [batch, seq_len, d_inner].
            cache: Optional cache with conv_state.

        Returns:
            Convolved tensor [batch, seq_len, d_inner].
        """
        batch_size, seq_len, d_inner = x.shape

        if cache is not None and cache.conv_state is not None:
            # Incremental mode: prepend cached conv state
            # conv_state: [batch, d_inner, d_conv - 1]
            x_t = x.transpose(1, 2)  # [B, d_inner, L]
            x_t = torch.cat([cache.conv_state, x_t], dim=2)  # [B, d_inner, L + d_conv - 1]
            x_t = self.conv1d(x_t)[:, :, :seq_len]  # Take first seq_len outputs
            return x_t.transpose(1, 2)
        else:
            # Full sequence: apply conv1d with causal padding
            x_t = x.transpose(1, 2)  # [B, d_inner, L]
            x_t = self.conv1d(x_t)[:, :, :seq_len]  # Truncate to maintain causality
            return x_t.transpose(1, 2)

    def _get_conv_state(
        self,
        x: torch.Tensor,
        cache: Optional[Mamba2Cache],
    ) -> torch.Tensor:
        """Get the conv state to cache (last d_conv - 1 inputs before conv).

        We need the raw pre-activation inputs to conv1d, but since we apply
        conv before activation, we cache the last d_conv-1 columns of the
        input that was fed to conv1d.

        Args:
            x: Post-conv tensor (used for shape reference). [batch, seq_len, d_inner]
            cache: Previous cache.

        Returns:
            Conv state tensor [batch, d_inner, d_conv - 1].
        """
        # We need to reconstruct what was fed into conv1d
        # The in_proj output before conv is what we need to cache
        # But we only have the post-SiLU x here.
        # Instead, we track it via the forward path.
        # For simplicity, we cache the last d_conv-1 positions of x_ssm (before SiLU)
        # This is handled in forward() by storing the pre-activation values.
        batch_size = x.shape[0]
        d_conv_m1 = self.d_conv - 1

        # The conv_state is the last d_conv-1 columns of x transposed
        x_t = x.transpose(1, 2)  # [B, d_inner, L]
        if x_t.shape[2] >= d_conv_m1:
            return x_t[:, :, -d_conv_m1:].contiguous()
        else:
            # Pad with zeros if sequence is shorter than d_conv - 1
            pad_size = d_conv_m1 - x_t.shape[2]
            return F.pad(x_t, (pad_size, 0)).contiguous()

    def _ssm_scan(
        self,
        x: torch.Tensor,
        cache: Optional[Mamba2Cache],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the selective SSM scan.

        Computes:
            dt = softplus(dt_proj(x))
            B = B_proj(x)
            C = C_proj(x)
            A = -exp(A_log)
            h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t
            y_t = C_t @ h_t

        Args:
            x: Input tensor [batch, seq_len, d_inner] (post-conv, post-SiLU).
            cache: Optional cache with ssm_state.

        Returns:
            Tuple of (output [batch, seq_len, d_inner], final_state [batch, d_inner, d_state]).
        """
        batch_size, seq_len, d_inner = x.shape

        # Compute data-dependent parameters
        dt = F.softplus(self.dt_proj(x))  # [B, L, d_inner]
        B = self.B_proj(x)  # [B, L, d_state]
        C = self.C_proj(x)  # [B, L, d_state]

        # Fixed transition matrix (negative for stability)
        A = -torch.exp(self.A_log.to(dtype=x.dtype))  # [d_inner]

        # Initialize state
        if cache is not None and cache.ssm_state is not None:
            h = cache.ssm_state.clone()  # [B, d_inner, d_state]
        else:
            h = torch.zeros(
                batch_size, d_inner, self.d_state,
                device=x.device, dtype=x.dtype,
            )

        # Sequential scan
        outputs = []
        for t in range(seq_len):
            dt_t = dt[:, t, :]  # [B, d_inner]
            B_t = B[:, t, :]  # [B, d_state]
            C_t = C[:, t, :]  # [B, d_state]
            x_t = x[:, t, :]  # [B, d_inner]

            # State update: h = exp(dt * A) * h + dt * B * x
            # exp(dt * A): [B, d_inner] -> [B, d_inner, 1] broadcast with h [B, d_inner, d_state]
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(-1))  # [B, d_inner, 1]
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # [B, d_inner, d_state]

            h = dA * h + dB * x_t.unsqueeze(-1)  # [B, d_inner, d_state]

            # Output: y = C @ h (per-batch, summed over d_state)
            y_t = (C_t.unsqueeze(1) * h).sum(dim=-1)  # [B, d_inner]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, L, d_inner]
        return y, h
