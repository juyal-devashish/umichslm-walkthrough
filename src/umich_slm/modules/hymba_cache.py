"""Hybrid cache for Hymba (attention + SSM state).

This module provides a cache implementation that stores both standard attention
KV cache and Mamba2 SSM recurrent state per layer. It supports cross-layer KV
sharing where adjacent layers reference the same attention KV entries.

References:
- Hymba: https://arxiv.org/abs/2411.13676
"""

from typing import Dict, List, Optional, Tuple

import torch

from .mamba2 import Mamba2Cache


def _build_kv_share_map(num_layers: int, group_size: int) -> Dict[int, int]:
    """Build mapping from layer_idx to its KV source layer_idx.

    Within each group, only the first layer (the "leader") computes fresh KV.
    Follower layers redirect to the leader's KV cache slot.

    Args:
        num_layers: Total number of layers.
        group_size: Number of consecutive layers per sharing group.

    Returns:
        Dict mapping each layer_idx to the layer_idx that owns its KV cache.
    """
    if group_size <= 1:
        return {i: i for i in range(num_layers)}

    share_map = {}
    for i in range(num_layers):
        leader = (i // group_size) * group_size
        share_map[i] = leader
    return share_map


class HymbaDynamicCache:
    """Dynamic cache for Hymba hybrid attention-SSM layers.

    Stores both attention KV cache and SSM state per layer. Supports cross-layer
    KV cache sharing where adjacent layers reference the same attention KV entries.

    Attributes:
        attn_key_cache: List of attention key tensors per layer.
        attn_value_cache: List of attention value tensors per layer.
        ssm_states: List of Mamba2Cache per layer (SSM recurrent + conv state).
        kv_share_map: Maps layer_idx -> KV source layer_idx.
    """

    def __init__(
        self,
        num_layers: int = 0,
        share_kv: bool = False,
        kv_share_group_size: int = 2,
    ):
        self.attn_key_cache: List[Optional[torch.Tensor]] = []
        self.attn_value_cache: List[Optional[torch.Tensor]] = []
        self.ssm_states: List[Optional[Mamba2Cache]] = []
        self._seen_tokens = 0
        self.num_layers = num_layers

        # Build KV sharing map
        if share_kv and num_layers > 0:
            self.kv_share_map = _build_kv_share_map(num_layers, kv_share_group_size)
        else:
            self.kv_share_map = {}

    @property
    def seen_tokens(self) -> int:
        """Return the number of tokens seen so far."""
        return self._seen_tokens

    def __len__(self) -> int:
        """Return the number of layers in the cache."""
        return max(len(self.attn_key_cache), len(self.ssm_states))

    def __getitem__(self, layer_idx: int):
        """Get cache for a specific layer. Returns self for HuggingFace compat."""
        return self

    def _get_kv_source(self, layer_idx: int) -> int:
        """Get the KV source layer for a given layer (for cross-layer sharing)."""
        if self.kv_share_map:
            return self.kv_share_map.get(layer_idx, layer_idx)
        return layer_idx

    def update_attn(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the attention KV cache for a layer.

        If cross-layer KV sharing is enabled, follower layers redirect their
        update to the group leader's slot.

        Args:
            key_states: New key tensor [batch, num_heads, seq_len, head_dim].
            value_states: New value tensor of same shape.
            layer_idx: Layer index.

        Returns:
            Tuple of (full_key_cache, full_value_cache) for this layer's source.
        """
        source_idx = self._get_kv_source(layer_idx)

        # Update seen tokens on first layer
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]

        # Ensure cache lists are long enough
        while len(self.attn_key_cache) <= source_idx:
            self.attn_key_cache.append(None)
            self.attn_value_cache.append(None)

        # Initialize or concatenate
        if self.attn_key_cache[source_idx] is None:
            self.attn_key_cache[source_idx] = key_states
            self.attn_value_cache[source_idx] = value_states
        else:
            self.attn_key_cache[source_idx] = torch.cat(
                [self.attn_key_cache[source_idx], key_states], dim=2
            )
            self.attn_value_cache[source_idx] = torch.cat(
                [self.attn_value_cache[source_idx], value_states], dim=2
            )

        return self.attn_key_cache[source_idx], self.attn_value_cache[source_idx]

    def set_attn_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Set (replace) the attention KV cache for a layer.

        Unlike update_attn which concatenates, this replaces the entire cache.
        Used when the inner attention module has already handled concatenation.

        Args:
            key_states: Full key tensor [batch, num_heads, seq_len, head_dim].
            value_states: Full value tensor of same shape.
            layer_idx: Layer index.
        """
        source_idx = self._get_kv_source(layer_idx)

        # Update seen tokens on first layer
        if layer_idx == 0:
            self._seen_tokens = key_states.shape[2]

        # Ensure cache lists are long enough
        while len(self.attn_key_cache) <= source_idx:
            self.attn_key_cache.append(None)
            self.attn_value_cache.append(None)

        self.attn_key_cache[source_idx] = key_states
        self.attn_value_cache[source_idx] = value_states

    def get_attn_cache(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get the attention KV cache for a layer (respects sharing)."""
        source_idx = self._get_kv_source(layer_idx)
        if source_idx >= len(self.attn_key_cache) or self.attn_key_cache[source_idx] is None:
            return None
        return (self.attn_key_cache[source_idx], self.attn_value_cache[source_idx])

    def update_ssm(
        self,
        ssm_cache: Mamba2Cache,
        layer_idx: int,
    ) -> None:
        """Update the SSM state for a layer. SSM states are NOT shared."""
        while len(self.ssm_states) <= layer_idx:
            self.ssm_states.append(None)
        self.ssm_states[layer_idx] = ssm_cache

    def get_ssm_cache(self, layer_idx: int) -> Optional[Mamba2Cache]:
        """Get the SSM cache for a layer."""
        if layer_idx >= len(self.ssm_states):
            return None
        return self.ssm_states[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the current sequence length stored in the attention cache."""
        source_idx = self._get_kv_source(layer_idx)
        if source_idx >= len(self.attn_key_cache) or self.attn_key_cache[source_idx] is None:
            return 0
        return self.attn_key_cache[source_idx].shape[2]

    def get_max_length(self) -> Optional[int]:
        """Dynamic cache has no maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder the cache for beam search."""
        for i in range(len(self.attn_key_cache)):
            if self.attn_key_cache[i] is not None:
                self.attn_key_cache[i] = self.attn_key_cache[i].index_select(0, beam_idx)
                self.attn_value_cache[i] = self.attn_value_cache[i].index_select(0, beam_idx)

        for i in range(len(self.ssm_states)):
            if self.ssm_states[i] is not None:
                ssm = self.ssm_states[i]
                if ssm.ssm_state is not None:
                    ssm.ssm_state = ssm.ssm_state.index_select(0, beam_idx)
                if ssm.conv_state is not None:
                    ssm.conv_state = ssm.conv_state.index_select(0, beam_idx)

    @classmethod
    def from_legacy_cache(
        cls, past_key_values, num_layers: int = 0
    ) -> Optional["HymbaDynamicCache"]:
        """Create from legacy cache format (for compatibility)."""
        if past_key_values is None:
            return None

        cache = cls(num_layers=num_layers)
        if isinstance(past_key_values, (list, tuple)):
            for layer_idx, layer_cache in enumerate(past_key_values):
                if layer_cache is None:
                    continue
                if isinstance(layer_cache, (list, tuple)) and len(layer_cache) >= 2:
                    cache.attn_key_cache.append(layer_cache[0])
                    cache.attn_value_cache.append(layer_cache[1])
                    cache.ssm_states.append(None)

        if cache.attn_key_cache:
            cache._seen_tokens = cache.attn_key_cache[0].shape[2]

        return cache
