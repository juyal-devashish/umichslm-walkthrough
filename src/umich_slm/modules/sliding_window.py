"""Sliding Window Attention with Global Refresh.

This module implements sliding window attention (SWA) with periodic global
refresh layers, as used in models like Mistral and OLMo-3.

Key concepts:
- Sliding Window: Each token attends only to a local window of nearby tokens
- Global Refresh: Every N layers uses full attention to propagate information globally
- Rolling Cache: For inference, maintain a fixed-size cache for memory efficiency

References:
    - Mistral 7B: https://arxiv.org/abs/2310.06825
    - Longformer: https://arxiv.org/abs/2004.05150
    - OLMo-3: https://arxiv.org/abs/2512.13961
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a sliding window causal attention mask.

    Each position can attend to at most `window_size` previous positions
    (including itself), maintaining causality.

    Args:
        seq_len: Sequence length.
        window_size: Size of the attention window.
        device: Device to create the mask on.
        dtype: Data type for the mask.

    Returns:
        Attention mask of shape (1, 1, seq_len, seq_len) where positions
        outside the window are masked with -inf.

    Example:
        For seq_len=5, window_size=3:
        [[0, -inf, -inf, -inf, -inf],
         [0,    0, -inf, -inf, -inf],
         [0,    0,    0, -inf, -inf],
         [-inf, 0,    0,    0, -inf],
         [-inf, -inf, 0,    0,    0]]
    """
    # Create position indices
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)

    # Causal mask: can only attend to previous positions
    causal_mask = col_idx <= row_idx

    # Window mask: can only attend within window
    # Position i can attend to positions max(0, i - window_size + 1) to i
    window_mask = (row_idx - col_idx) < window_size

    # Combine masks
    mask = causal_mask & window_mask

    # Convert to attention mask format (-inf for masked positions)
    attention_mask = torch.where(
        mask,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )

    # Add batch and head dimensions
    return attention_mask.unsqueeze(0).unsqueeze(0)


def create_sliding_window_mask_for_sdpa(
    seq_len: int,
    window_size: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create sliding window mask compatible with scaled_dot_product_attention.

    SDPA expects boolean mask where True means "do not attend".

    Args:
        seq_len: Sequence length.
        window_size: Size of the attention window.
        device: Device to create the mask on.
        dtype: Data type (for compatibility, returns bool).

    Returns:
        Boolean attention mask of shape (1, 1, seq_len, seq_len).
    """
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)

    # Causal: can attend to col <= row
    causal_mask = col_idx <= row_idx

    # Window: can attend within window
    window_mask = (row_idx - col_idx) < window_size

    # Combined: True where we CAN attend
    can_attend = causal_mask & window_mask

    # SDPA wants True where we CANNOT attend
    return (~can_attend).unsqueeze(0).unsqueeze(0)


def update_sliding_window_mask_for_cache(
    attention_mask: torch.Tensor,
    query_len: int,
    key_len: int,
    window_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Update attention mask for cached inference with sliding window.

    During generation with KV cache, we need to create a mask that:
    1. Allows the new query tokens to attend to the cached keys within the window
    2. Maintains the sliding window constraint

    Args:
        attention_mask: Input attention mask from user (batch, key_len) or None.
        query_len: Number of new query tokens.
        key_len: Total length of key/value cache.
        window_size: Size of the attention window.
        dtype: Data type for the mask.

    Returns:
        Attention mask of shape (1, 1, query_len, key_len).
    """
    device = attention_mask.device if attention_mask is not None else None

    # For each query position, determine which key positions are within window
    # Query position i (in full sequence) = key_len - query_len + i
    # Key position j is within window if: query_pos - j < window_size

    query_positions = torch.arange(
        key_len - query_len, key_len, device=device
    ).unsqueeze(1)  # [query_len, 1]
    key_positions = torch.arange(key_len, device=device).unsqueeze(0)  # [1, key_len]

    # Causal: can attend to key positions <= query position
    causal_mask = key_positions <= query_positions

    # Window: distance must be < window_size
    window_mask = (query_positions - key_positions) < window_size

    # Combine
    can_attend = causal_mask & window_mask

    # Convert to float mask
    mask = torch.where(
        can_attend,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )

    # Apply user attention mask if provided
    if attention_mask is not None:
        # attention_mask is typically (batch, key_len) with 1s and 0s
        if attention_mask.dim() == 2:
            user_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, key_len]
            user_mask = torch.where(
                user_mask == 0,
                torch.tensor(float("-inf"), dtype=dtype, device=device),
                torch.tensor(0.0, dtype=dtype, device=device),
            )
            mask = mask + user_mask

    return mask.unsqueeze(0).unsqueeze(0) if mask.dim() == 2 else mask


# Attention kinds in a `hybrid_attention_ratio` schedule that denote the periodic
# FULL-context refresh layer. Everything else is banded. This set is kept in sync
# with the FLOP model in `titan/throughput.py:143`, which already bills exactly
# ("mla", "attn", "delta") at full causal cost -- so a kind billed as global but
# executed as windowed is both an architecture bug and an MFU misreport.
# "attn" is the plain-GQA refresh layer of the [delta, delta, delta, attn]
# qwen3.5-style schedules; "delta" is excluded because a recurrent layer has no
# window to lift in the first place.
GLOBAL_ATTENTION_KINDS = frozenset({"mla", "attn", "global", "full"})


def is_global_attention_layer(
    layer_idx: int,
    global_attention_every_n_layers: int,
    hybrid_ratio: Optional[Sequence[str]] = None,
) -> bool:
    """Determine if a layer should use global (full) attention.

    In the global refresh pattern, every N-th layer uses full attention
    to allow information to propagate beyond the local window.

    When ``hybrid_ratio`` (the expanded per-layer attention schedule) is given, the
    global layer is the one the SCHEDULE designates -- the ``mla`` entry of a
    ``[swa, swa, swa, mla]`` pattern -- rather than an index derived independently.

    This matters because the two disagreed silently. The 3:1 pattern puts MLA at
    0-indexed 3/7/11 while the modulo rule below marks 0/4/8, so **no MLA layer
    was ever global** and the architecture had no full-context layer at all,
    contradicting `candidate_architectures.tex` §2. Re-phasing the modulo would
    not fix it in general either: `_ratio_final_global` forces the LAST layer to
    the global kind, which for e.g. L=10 lands at index 9 and no `% 4` rule
    catches it. Reading the schedule is the only rule that cannot drift from it.

    Args:
        layer_idx: Index of the current layer (0-based).
        global_attention_every_n_layers: Frequency of global attention layers.
            If 0 or negative, all layers use sliding window. Ignored when
            ``hybrid_ratio`` is supplied.
        hybrid_ratio: Optional per-layer attention schedule.

    Returns:
        True if this layer should use global attention.

    Examples:
        >>> is_global_attention_layer(3, 4, ["swa", "swa", "swa", "mla"])
        True
        >>> is_global_attention_layer(0, 4, ["swa", "swa", "swa", "mla"])
        False
        >>> is_global_attention_layer(0, 4)  # no schedule: legacy modulo rule
        True
        >>> is_global_attention_layer(7, 4, ["swa", "swa", "swa", "mla"])  # cycles
        True
    """
    if hybrid_ratio:
        # A ratio SHORTER than num_hidden_layers is a repeating pattern, and both
        # other consumers already read it that way: `_ratio_entry_for_layer`
        # (transformer_block.py) picks the layer's attention CLASS with
        # `pattern[idx % len(pattern)]`, and `per_layer_avg_keys`
        # (titan/throughput.py) BILLS it with the same modulo. Truncating here
        # instead of cycling would rebuild the exact bug this function exists to
        # kill, just past the end of the pattern: the layer gets an MLA module
        # and is billed at full causal cost, but executes windowed. That is live
        # for the deep stacking presets, which carry a bare 4-cycle at L=48..80
        # (stack_100m_b: 19 of 20 MLA layers). The sweep arms are unaffected only
        # because `_ratio_final_global` pre-expands their pattern to full depth.
        return str(hybrid_ratio[layer_idx % len(hybrid_ratio)]).lower() in GLOBAL_ATTENTION_KINDS

    if global_attention_every_n_layers <= 0:
        return False

    # First layer (index 0) is typically global
    if layer_idx == 0:
        return True

    # Every N-th layer is global
    return (layer_idx % global_attention_every_n_layers) == 0


class SlidingWindowCache:
    """Rolling KV cache for sliding window attention.

    This cache maintains only the most recent `window_size` tokens,
    discarding older tokens to maintain constant memory usage during generation.

    This is more memory-efficient than full caching for long sequences
    when using sliding window attention.

    Args:
        window_size: Maximum number of tokens to cache.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        batch_size: Batch size.
        dtype: Data type for cached tensors.
        device: Device for cached tensors.
    """

    def __init__(
        self,
        window_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        self.window_size = window_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device

        # Initialize empty cache for each layer
        # Each entry is (key_cache, value_cache) with shape [batch, heads, window, head_dim]
        self._cache = {}
        self._seq_lengths = {}  # Track actual sequence length per layer

        for layer_idx in range(num_layers):
            self._cache[layer_idx] = (
                torch.zeros(
                    batch_size, num_heads, window_size, head_dim,
                    dtype=dtype, device=device
                ),
                torch.zeros(
                    batch_size, num_heads, window_size, head_dim,
                    dtype=dtype, device=device
                ),
            )
            self._seq_lengths[layer_idx] = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key/value states.

        If the cache is full, old tokens are rolled out.

        Args:
            key_states: New key states of shape [batch, heads, new_len, head_dim].
            value_states: New value states of shape [batch, heads, new_len, head_dim].
            layer_idx: Index of the layer.

        Returns:
            Tuple of (cached_keys, cached_values) containing all cached states
            up to window_size.
        """
        new_len = key_states.shape[2]
        current_len = self._seq_lengths[layer_idx]
        key_cache, value_cache = self._cache[layer_idx]

        if current_len + new_len <= self.window_size:
            # Cache not full, just append
            key_cache[:, :, current_len:current_len + new_len] = key_states
            value_cache[:, :, current_len:current_len + new_len] = value_states
            self._seq_lengths[layer_idx] = current_len + new_len
        else:
            # Cache will overflow, need to roll
            total_len = current_len + new_len
            keep_len = min(self.window_size, total_len)
            discard_len = total_len - keep_len

            if discard_len >= current_len:
                # All old tokens discarded, just take newest
                new_start = new_len - keep_len
                key_cache[:, :, :keep_len] = key_states[:, :, new_start:]
                value_cache[:, :, :keep_len] = value_states[:, :, new_start:]
            else:
                # Keep some old tokens
                old_keep = current_len - discard_len
                # Shift old tokens left
                key_cache[:, :, :old_keep] = key_cache[:, :, discard_len:current_len].clone()
                value_cache[:, :, :old_keep] = value_cache[:, :, discard_len:current_len].clone()
                # Add new tokens
                key_cache[:, :, old_keep:old_keep + new_len] = key_states
                value_cache[:, :, old_keep:old_keep + new_len] = value_states

            self._seq_lengths[layer_idx] = keep_len

        # Return the active portion of the cache
        seq_len = self._seq_lengths[layer_idx]
        return key_cache[:, :, :seq_len], value_cache[:, :, :seq_len]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the current sequence length in the cache."""
        return self._seq_lengths.get(layer_idx, 0)

    def reset(self):
        """Reset the cache to empty state."""
        for layer_idx in range(self.num_layers):
            self._cache[layer_idx][0].zero_()
            self._cache[layer_idx][1].zero_()
            self._seq_lengths[layer_idx] = 0


def get_attention_pattern_for_layer(
    layer_idx: int,
    num_layers: int,
    window_size: int,
    global_attention_every_n_layers: int,
    attention_pattern: str = "sliding_window_global_refresh",
) -> str:
    """Determine the attention pattern for a specific layer.

    Args:
        layer_idx: Index of the current layer.
        num_layers: Total number of layers.
        window_size: Sliding window size.
        global_attention_every_n_layers: Frequency of global attention layers.
        attention_pattern: Base attention pattern type.

    Returns:
        "global" for full attention, "local" for sliding window.
    """
    if attention_pattern == "full":
        return "global"
    elif attention_pattern == "sliding_window":
        return "local"
    elif attention_pattern == "sliding_window_global_refresh":
        if is_global_attention_layer(layer_idx, global_attention_every_n_layers):
            return "global"
        return "local"
    else:
        # Default to global
        return "global"
