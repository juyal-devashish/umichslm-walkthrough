"""Mixture-of-Depths Attention (MoDA) module.

MoDA lets each attention head jointly attend to sequence KV pairs (standard)
AND depth KV pairs from all preceding layers, under a single unified softmax.
This combats signal degradation in deep networks where informative features
from shallow layers get diluted by repeated residual updates.

Reference: MoDA: Mixture-of-Depths Attention (hustvl/MoDA)
"""

import logging
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MoDADepthKVCache:
    """Cache that stores depth KV pairs from preceding layers.

    Each entry is a (K_depth, V_depth) pair from a layer's output,
    used by subsequent layers to attend across depth.

    Args:
        max_entries: Maximum depth entries to keep (-1 = unlimited).
        depth_kv_every_n: Only store depth KV from every Nth layer (1 = all).
        active_layers: Set of layer indices that participate in MoDA.
            If None, all layers participate.
    """

    def __init__(
        self,
        max_entries: int = -1,
        depth_kv_every_n: int = 1,
        active_layers: Optional[Set[int]] = None,
    ):
        self.max_entries = max_entries
        self.depth_kv_every_n = depth_kv_every_n
        self.active_layers = active_layers
        # List of (K, V) tuples, each of shape (batch, num_kv_heads, seq_len, head_dim)
        self._cache: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Add a layer's depth KV to the cache.

        Only stores if the layer passes the every-N filter and is in active_layers.

        Args:
            layer_idx: Index of the layer producing this KV.
            k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim).
            v: Value tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        """
        if self.active_layers is not None and layer_idx not in self.active_layers:
            return
        if self.depth_kv_every_n > 1 and layer_idx % self.depth_kv_every_n != 0:
            return

        self._cache.append((k, v))

        # Truncate if max_entries exceeded (FIFO eviction)
        if self.max_entries > 0 and len(self._cache) > self.max_entries:
            self._cache = self._cache[-self.max_entries :]

    def get_depth_kv(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get concatenated depth KV from all entries before this layer.

        Args:
            layer_idx: Current layer index (only entries before this are returned).

        Returns:
            Tuple of (K_depth, V_depth) concatenated along the sequence dimension,
            or None if no depth KV entries exist.
            Shape: (batch, num_kv_heads, num_entries * seq_len, head_dim)
        """
        if self.active_layers is not None and layer_idx not in self.active_layers:
            return None
        if len(self._cache) == 0:
            return None

        keys = [entry[0] for entry in self._cache]
        values = [entry[1] for entry in self._cache]

        # Concatenate along the sequence dimension (dim=2)
        k_depth = torch.cat(keys, dim=2)
        v_depth = torch.cat(values, dim=2)

        return k_depth, v_depth

    def reset(self):
        """Clear the cache."""
        self._cache.clear()

    def __len__(self) -> int:
        """Number of entries in the cache."""
        return len(self._cache)


class MoDAFFNProjection(nn.Module):
    """Project FFN input to depth KV pairs.

    Lightweight linear projections that produce K and V from the FFN input,
    providing additional depth information for subsequent layers.

    Args:
        hidden_size: Model hidden dimension.
        num_kv_heads: Number of key-value heads.
        head_dim: Dimension per head.
    """

    def __init__(self, hidden_size: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project hidden states to depth K, V.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Tuple of (K, V) each of shape (batch, num_kv_heads, seq_len, head_dim).
        """
        bsz, seq_len, _ = hidden_states.shape
        k = (
            self.k_proj(hidden_states)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        return k, v


def build_moda_combined_mask(
    seq_len: int,
    depth_seq_len: int,
    num_depth_layers: int,
    device: torch.device,
    dtype: torch.dtype,
    past_len: int = 0,
) -> torch.Tensor:
    """Build a combined attention mask for sequence + depth KV.

    The mask has two parts:
    1. Causal mask for sequence KV (standard causal attention)
    2. Block-diagonal mask for depth KV: token at position t can only attend
       to depth entries from its own position across layers.

    The depth KV is organized as [layer0_pos0, layer0_pos1, ..., layer1_pos0, ...]
    i.e., num_depth_layers blocks of seq_len entries each.

    Args:
        seq_len: Query sequence length.
        depth_seq_len: Total depth KV sequence length (num_entries * original_seq_len).
        num_depth_layers: Number of depth layers in the cache.
        device: Torch device.
        dtype: Torch dtype for the mask.
        past_len: Length of past cached sequence (for incremental decoding).

    Returns:
        Mask tensor of shape (1, 1, seq_len, total_kv_len) where
        total_kv_len = (past_len + seq_len) + depth_seq_len.
        Masked positions are -inf, unmasked are 0.
    """
    total_seq_kv_len = past_len + seq_len
    total_kv_len = total_seq_kv_len + depth_seq_len

    # Start with all masked
    mask = torch.full((seq_len, total_kv_len), float("-inf"), device=device, dtype=dtype)

    # Part 1: Causal mask for sequence KV
    for i in range(seq_len):
        # Can attend to all past positions + current and previous positions
        mask[i, : past_len + i + 1] = 0.0

    # Part 2: Block-diagonal mask for depth KV
    # Each depth entry at position t from any layer can be attended to by query at position t
    if depth_seq_len > 0 and num_depth_layers > 0:
        orig_seq_len = depth_seq_len // num_depth_layers
        for i in range(seq_len):
            # Query position i (accounting for past_len offset)
            query_pos = past_len + i
            # This query can attend to depth entries at the same token position
            # across all layers. For each layer l, the depth entry for position t
            # is at offset: total_seq_kv_len + l * orig_seq_len + t
            if query_pos < orig_seq_len:
                for layer_i in range(num_depth_layers):
                    depth_offset = total_seq_kv_len + layer_i * orig_seq_len + query_pos
                    if depth_offset < total_kv_len:
                        mask[i, depth_offset] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, total_kv_len)
