"""RePo (Re-Positioning) module for learned position embeddings.

This module implements the position re-positioning mechanism from:
"RePo: Language Models with Context Re-Positioning" (arXiv:2512.14391)

Instead of using fixed sequential position indices, RePo learns to predict
continuous position values from the hidden states, allowing the model to
learn semantically-informed position assignments.

The mechanism consists of:
1. Position Representation: r_i = Swish(h_i W^g) ⊙ (h_i W^c)  (SwiGLU)
2. Position Assignment: z_i = r_i W^z  (linear projection to per-head positions)

Reference: https://arxiv.org/abs/2512.14391
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RePoPositionPredictor(nn.Module):
    """Predicts continuous position values from hidden states.

    This module takes normalized hidden states and predicts continuous position
    values that replace the standard integer position indices in RoPE.

    The architecture follows the paper:
    - SwiGLU layer to extract position representation (shared across heads)
    - Linear projection to per-head position values

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        repo_dim: Dimension of position representation. Defaults to hidden_size // 8.
        bias: Whether to use bias in linear layers. Default False.

    Example:
        >>> predictor = RePoPositionPredictor(hidden_size=1024, num_heads=16)
        >>> hidden_states = torch.randn(2, 128, 1024)  # (batch, seq_len, hidden)
        >>> positions = predictor(hidden_states)  # (batch, seq_len, num_heads)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        repo_dim: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.repo_dim = repo_dim if repo_dim is not None else hidden_size // 8

        # SwiGLU components for position representation
        # r_i = Swish(h_i W^g) ⊙ (h_i W^c)
        self.gate_proj = nn.Linear(hidden_size, self.repo_dim, bias=bias)
        self.content_proj = nn.Linear(hidden_size, self.repo_dim, bias=bias)

        # Position assignment: z_i = r_i W^z
        # Projects to per-head position values
        self.position_proj = nn.Linear(self.repo_dim, num_heads, bias=bias)

        # Initialize position projection with small weights for stability
        # This ensures initial positions are close to standard sequential positions
        nn.init.normal_(self.position_proj.weight, mean=0.0, std=0.01)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict continuous positions from hidden states.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
                These should be the normalized hidden states before Q/K projection.

        Returns:
            positions: Continuous position values of shape (batch, seq_len, num_heads).
                These replace integer position indices in the RoPE computation.
        """
        # Position representation via SwiGLU
        # r_i = Swish(h_i W^g) ⊙ (h_i W^c)
        gate = F.silu(self.gate_proj(hidden_states))  # Swish = SiLU
        content = self.content_proj(hidden_states)
        representation = gate * content  # Element-wise multiplication

        # Position assignment
        # z_i = r_i W^z -> (batch, seq_len, num_heads)
        positions = self.position_proj(representation)

        return positions


class RePoPositionPredictorWithBase(nn.Module):
    """RePo predictor that adds learned offsets to base sequential positions.

    This variant predicts position *deltas* that are added to the standard
    sequential positions, which can help with training stability.

    positions = base_positions + scale * predicted_deltas

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        repo_dim: Dimension of position representation. Defaults to hidden_size // 8.
        position_scale: Scale factor for predicted deltas. Default 1.0.
        bias: Whether to use bias in linear layers. Default False.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        repo_dim: Optional[int] = None,
        position_scale: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.repo_dim = repo_dim if repo_dim is not None else hidden_size // 8
        self.position_scale = position_scale

        # SwiGLU components
        self.gate_proj = nn.Linear(hidden_size, self.repo_dim, bias=bias)
        self.content_proj = nn.Linear(hidden_size, self.repo_dim, bias=bias)

        # Position delta projection
        self.delta_proj = nn.Linear(self.repo_dim, num_heads, bias=bias)

        # Initialize with very small weights so initial deltas are near zero
        nn.init.normal_(self.delta_proj.weight, mean=0.0, std=0.001)

    def forward(
        self,
        hidden_states: torch.Tensor,
        base_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict positions as base + learned deltas.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
            base_position_ids: Optional base positions of shape (batch, seq_len).
                If None, uses sequential positions [0, 1, 2, ...].

        Returns:
            positions: Continuous position values of shape (batch, seq_len, num_heads).
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Get base sequential positions
        if base_position_ids is None:
            base_positions = torch.arange(seq_len, device=device, dtype=hidden_states.dtype)
            base_positions = base_positions.unsqueeze(0).expand(batch_size, -1)
        else:
            base_positions = base_position_ids.to(hidden_states.dtype)

        # Expand to per-head: (batch, seq_len) -> (batch, seq_len, num_heads)
        base_positions = base_positions.unsqueeze(-1).expand(-1, -1, self.num_heads)

        # Predict deltas via SwiGLU
        gate = F.silu(self.gate_proj(hidden_states))
        content = self.content_proj(hidden_states)
        representation = gate * content

        # Position deltas
        deltas = self.delta_proj(representation)

        # Combine: base + scaled deltas
        positions = base_positions + self.position_scale * deltas
        positions = positions.clamp(min=0.0)

        return positions
