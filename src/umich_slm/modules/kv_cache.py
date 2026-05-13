"""KV Cache implementation for efficient autoregressive generation.

This module provides cache implementations for storing key-value pairs
during generation to avoid recomputing attention for previous tokens.
"""

from typing import Dict, Iterator, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache

from .deltanet import DeltaNetCache


class DynamicCache:
    """Dynamic KV cache that grows with generation.

    This cache stores key and value tensors for each layer, allowing
    efficient autoregressive generation by avoiding recomputation of
    attention for previously generated tokens.

    Attributes:
        key_cache: List of key tensors, one per layer.
        value_cache: List of value tensors, one per layer.
        seen_tokens: Total number of tokens processed.

    Shape:
        - Key/Value tensors: (batch_size, num_heads, seq_len, head_dim)
    """

    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    @property
    def seen_tokens(self) -> int:
        """Return the number of tokens seen so far."""
        return self._seen_tokens

    def __len__(self) -> int:
        """Return the number of layers in the cache."""
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the cache with new key and value states.

        Args:
            key_states: New key tensor of shape (batch, num_heads, seq_len, head_dim).
            value_states: New value tensor of same shape.
            layer_idx: The layer index to update.

        Returns:
            Tuple of (full_key_cache, full_value_cache) for this layer,
            concatenated with previously cached values.
        """
        # Update seen tokens count on first layer
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]

        # Initialize cache for this layer if needed
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Concatenate new states with cached states
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the current sequence length stored in the cache.

        Args:
            layer_idx: Which layer's cache to check.

        Returns:
            The sequence length in the cache for the specified layer.
        """
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[2]

    def get_max_length(self) -> Optional[int]:
        """Get the maximum length of the cache.

        Returns:
            None (dynamic cache has no maximum length).
        """
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder the cache for beam search.

        Args:
            beam_idx: Tensor of beam indices to select.
        """
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Convert to legacy cache format (tuple of tuples).

        Returns:
            Tuple of (key, value) tuples for each layer.
        """
        return tuple(
            (self.key_cache[layer_idx], self.value_cache[layer_idx])
            for layer_idx in range(len(self.key_cache))
        )

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]
    ) -> Optional["DynamicCache"]:
        """Create a DynamicCache from legacy cache format.

        Args:
            past_key_values: Legacy cache format (tuple of tuples).

        Returns:
            DynamicCache instance or None if input is None.
        """
        if past_key_values is None:
            return None

        cache = cls()
        for layer_idx, (key_states, value_states) in enumerate(past_key_values):
            cache.key_cache.append(key_states)
            cache.value_cache.append(value_states)

        if cache.key_cache:
            cache._seen_tokens = cache.key_cache[0].shape[2]

        return cache


class _AttentionLayerCacheView:
    """Per-layer attention cache view backed by DeltaNetDynamicCache."""

    def __init__(self, parent: "DeltaNetDynamicCache", layer_idx: int):
        self.parent = parent
        self.layer_idx = layer_idx

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update this layer's KV cache and return the full cached tensors."""
        del layer_idx  # The view is already bound to a single layer.
        return self.parent.update_attention(key_states, value_states, self.layer_idx)

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """Get the cached sequence length for this layer."""
        del layer_idx
        return self.parent.get_seq_length(self.layer_idx)

    def get_max_length(self) -> Optional[int]:
        """Return the cache max length."""
        return self.parent.get_max_length()


class DeltaNetDynamicCache(Cache):
    """Shared generation cache for pure/hybrid DeltaNet models.

    This cache stores standard KV states for attention layers and DeltaNetCache
    objects for DeltaNet layers. It presents a cache-like interface to
    generation while still allowing each layer to receive the cache shape it
    expects.
    """

    def __init__(self, num_layers: int, deltanet_layers: List[bool]):
        # Cache.__init__ changed across Transformers releases. Newer versions
        # accept `layers=[]`, while older supported releases in this repo do
        # not expose that kwarg. Keep this constructor compatible with both.
        try:
            super().__init__(layers=[])
        except TypeError:
            try:
                super().__init__()
            except TypeError:
                pass
            if not hasattr(self, "layers"):
                self.layers = []
        self.num_layers = num_layers
        self.deltanet_layers = list(deltanet_layers)
        self.key_cache: Dict[int, torch.Tensor] = {}
        self.value_cache: Dict[int, torch.Tensor] = {}
        self.deltanet_cache: List[Optional[DeltaNetCache]] = [None] * num_layers
        self._seen_tokens = 0
        self._attention_views = {
            layer_idx: _AttentionLayerCacheView(self, layer_idx)
            for layer_idx, uses_delta in enumerate(self.deltanet_layers)
            if not uses_delta
        }

    def __len__(self) -> int:
        return self.num_layers

    def __iter__(self) -> Iterator[Optional[object]]:
        for layer_idx in range(self.num_layers):
            yield self[layer_idx]

    def __getitem__(self, layer_idx: int):
        if self.deltanet_layers[layer_idx]:
            return self.deltanet_cache[layer_idx]
        return self._attention_views[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache state using the standard Cache interface."""
        del args, kwargs
        return self.update_attention(key_states, value_states, layer_idx)

    def update_attention(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache for one attention layer."""
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def set_deltanet_cache(self, layer_idx: int, cache: Optional[DeltaNetCache]) -> None:
        """Store DeltaNet recurrent/conv state for one layer."""
        if cache is not None and cache.seen_tokens is None:
            cache = DeltaNetCache(
                recurrent_state=cache.recurrent_state,
                conv_state=cache.conv_state,
                seen_tokens=self._seen_tokens,
            )
        self.deltanet_cache[layer_idx] = cache

    def advance_seq_length(self, num_tokens: int) -> None:
        """Record how many new tokens were processed in this forward step."""
        self._seen_tokens += num_tokens

    @property
    def seen_tokens(self) -> int:
        """Return the number of tokens processed so far."""
        return self._seen_tokens

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the cached sequence length.

        Falls back to the globally tracked token count for DeltaNet layers.
        """
        if layer_idx in self.key_cache:
            return self.key_cache[layer_idx].shape[2]
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Dynamic cache has no fixed maximum length."""
        return None

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Dynamic cache has no fixed maximum length."""
        del layer_idx
        return -1

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> Tuple[int, int]:
        """Return mask sizes compatible with dynamic caches."""
        return self.get_seq_length(layer_idx) + query_length, 0

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder all cached states for beam search."""
        for layer_idx in list(self.key_cache.keys()):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)

        for layer_idx, cache in enumerate(self.deltanet_cache):
            if cache is None:
                continue

            recurrent_state = cache.recurrent_state
            conv_state = cache.conv_state
            if recurrent_state is not None:
                recurrent_state = recurrent_state.index_select(0, beam_idx)
            if conv_state is not None:
                conv_state = conv_state.index_select(0, beam_idx)
            self.deltanet_cache[layer_idx] = DeltaNetCache(
                recurrent_state=recurrent_state,
                conv_state=conv_state,
                seen_tokens=cache.seen_tokens,
            )

    def batch_repeat_interleave(self, repeats: int):
        """Repeat cache batch entries for contrastive search style generation."""
        for layer_idx in list(self.key_cache.keys()):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )

        for layer_idx, cache in enumerate(self.deltanet_cache):
            if cache is None:
                continue
            recurrent_state = cache.recurrent_state
            conv_state = cache.conv_state
            if recurrent_state is not None:
                recurrent_state = recurrent_state.repeat_interleave(repeats, dim=0)
            if conv_state is not None:
                conv_state = conv_state.repeat_interleave(repeats, dim=0)
            self.deltanet_cache[layer_idx] = DeltaNetCache(
                recurrent_state=recurrent_state,
                conv_state=conv_state,
                seen_tokens=cache.seen_tokens,
            )

    def batch_select_indices(self, indices: torch.Tensor):
        """Select a subset of the batch dimension in place."""
        for layer_idx in list(self.key_cache.keys()):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, indices)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, indices)

        for layer_idx, cache in enumerate(self.deltanet_cache):
            if cache is None:
                continue
            recurrent_state = cache.recurrent_state
            conv_state = cache.conv_state
            if recurrent_state is not None:
                recurrent_state = recurrent_state.index_select(0, indices)
            if conv_state is not None:
                conv_state = conv_state.index_select(0, indices)
            self.deltanet_cache[layer_idx] = DeltaNetCache(
                recurrent_state=recurrent_state,
                conv_state=conv_state,
                seen_tokens=cache.seen_tokens,
            )

    def crop(self, max_length: int):
        """Crop attention-cache tensors and tracked length in place."""
        if max_length < 0:
            max_length = max(self._seen_tokens + max_length, 0)

        for layer_idx in list(self.key_cache.keys()):
            if max_length == 0:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, :0, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, :0, :]
            else:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, -max_length:, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, -max_length:, :]

        self._seen_tokens = min(self._seen_tokens, max_length)
        for layer_idx, cache in enumerate(self.deltanet_cache):
            if cache is None:
                continue
            self.deltanet_cache[layer_idx] = DeltaNetCache(
                recurrent_state=cache.recurrent_state,
                conv_state=cache.conv_state,
                seen_tokens=self._seen_tokens,
            )

    def to_legacy_cache(self) -> Tuple[Optional[object], ...]:
        """Convert to the legacy per-layer tuple/object cache format."""
        legacy_cache = []
        for layer_idx in range(self.num_layers):
            if self.deltanet_layers[layer_idx]:
                cache = self.deltanet_cache[layer_idx]
                if cache is None:
                    legacy_cache.append(None)
                else:
                    legacy_cache.append(
                        DeltaNetCache(
                            recurrent_state=cache.recurrent_state,
                            conv_state=cache.conv_state,
                            seen_tokens=cache.seen_tokens,
                        )
                    )
            else:
                key_states = self.key_cache.get(layer_idx)
                value_states = self.value_cache.get(layer_idx)
                legacy_cache.append(
                    None if key_states is None or value_states is None else (key_states, value_states)
                )
        return tuple(legacy_cache)

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple[Optional[object], ...]],
        num_layers: int,
        deltanet_layers: List[bool],
    ) -> Optional["DeltaNetDynamicCache"]:
        """Create a DeltaNetDynamicCache from a legacy per-layer cache tuple."""
        if past_key_values is None:
            return None
        if isinstance(past_key_values, cls):
            return past_key_values

        cache = cls(num_layers=num_layers, deltanet_layers=deltanet_layers)
        for layer_idx, layer_cache in enumerate(past_key_values[:num_layers]):
            if layer_cache is None:
                continue
            if deltanet_layers[layer_idx]:
                if not isinstance(layer_cache, DeltaNetCache):
                    raise TypeError(
                        f"Expected DeltaNetCache for DeltaNet layer {layer_idx}, got "
                        f"{type(layer_cache).__name__}"
                    )
                cache.deltanet_cache[layer_idx] = layer_cache
                if layer_cache.seen_tokens is not None:
                    cache._seen_tokens = max(cache._seen_tokens, layer_cache.seen_tokens)
            else:
                if not isinstance(layer_cache, tuple) or len(layer_cache) != 2:
                    raise TypeError(
                        f"Expected (key, value) tuple for attention layer {layer_idx}, got "
                        f"{type(layer_cache).__name__}"
                    )
                key_states, value_states = layer_cache
                cache.key_cache[layer_idx] = key_states
                cache.value_cache[layer_idx] = value_states
                cache._seen_tokens = max(cache._seen_tokens, key_states.shape[2])

        return cache


class StaticCache:
    """Static KV cache with pre-allocated memory.

    This cache pre-allocates memory for a fixed maximum sequence length,
    avoiding dynamic memory allocation during generation. This can be
    more efficient for known maximum sequence lengths.

    Args:
        batch_size: Batch size.
        max_seq_len: Maximum sequence length to support.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        num_layers: Number of transformer layers.
        dtype: Data type for the cache tensors.
        device: Device for the cache tensors.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        cache_shape = (batch_size, num_heads, max_seq_len, head_dim)

        # Pre-allocate cache tensors
        self.key_cache = [
            torch.zeros(cache_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.value_cache = [
            torch.zeros(cache_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the cache with new key and value states.

        Args:
            key_states: New key tensor.
            value_states: New value tensor.
            layer_idx: The layer index to update.

        Returns:
            Tuple of (key_cache_slice, value_cache_slice) up to current position.
        """
        seq_len = key_states.shape[2]
        start_pos = self._seen_tokens if layer_idx == 0 else self.get_seq_length() - seq_len

        if layer_idx == 0:
            self._seen_tokens += seq_len

        # Copy new states into pre-allocated cache
        self.key_cache[layer_idx][:, :, start_pos : start_pos + seq_len, :] = key_states
        self.value_cache[layer_idx][:, :, start_pos : start_pos + seq_len, :] = value_states

        # Return the valid portion of the cache
        end_pos = start_pos + seq_len
        return (
            self.key_cache[layer_idx][:, :, :end_pos, :],
            self.value_cache[layer_idx][:, :, :end_pos, :],
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the current sequence length in the cache."""
        return self._seen_tokens

    def get_max_length(self) -> int:
        """Get the maximum length of the cache."""
        return self.max_seq_len

    def reset(self):
        """Reset the cache for a new generation."""
        self._seen_tokens = 0


class MLADynamicCache:
    """Dynamic cache for Multi-head Latent Attention (MLA).

    MLA uses a different cache format than standard attention:
    - c_kv: Compressed KV latent of shape (batch, seq_len, kv_lora_rank)
    - k_rope: RoPE-encoded keys of shape (batch, seq_len, num_heads, qk_rope_head_dim)

    This cache provides the same interface as DynamicCache so HuggingFace's
    generation utilities can work with MLA models.

    Attributes:
        c_kv_cache: List of compressed KV latent tensors, one per layer.
        k_rope_cache: List of RoPE key tensors, one per layer.
        seen_tokens: Total number of tokens processed.
    """

    def __init__(self):
        self.c_kv_cache: List[torch.Tensor] = []
        self.k_rope_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    @property
    def seen_tokens(self) -> int:
        """Return the number of tokens seen so far."""
        return self._seen_tokens

    def __len__(self) -> int:
        """Return the number of layers in the cache."""
        return len(self.c_kv_cache)

    def __getitem__(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cache for a specific layer."""
        if layer_idx >= len(self.c_kv_cache):
            return None
        return (self.c_kv_cache[layer_idx], self.k_rope_cache[layer_idx])

    def update(
        self,
        c_kv: torch.Tensor,
        k_rope: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the cache with new c_kv and k_rope states.

        Args:
            c_kv: New compressed KV tensor of shape (batch, seq_len, kv_lora_rank).
            k_rope: New RoPE key tensor of shape (batch, seq_len, num_heads, qk_rope_head_dim).
            layer_idx: The layer index to update.

        Returns:
            Tuple of (full_c_kv_cache, full_k_rope_cache) for this layer,
            concatenated with previously cached values.
        """
        # Update seen tokens count on first layer
        if layer_idx == 0:
            self._seen_tokens += c_kv.shape[1]

        # Initialize cache for this layer if needed
        if len(self.c_kv_cache) <= layer_idx:
            self.c_kv_cache.append(c_kv)
            self.k_rope_cache.append(k_rope)
        else:
            # Concatenate new states with cached states
            self.c_kv_cache[layer_idx] = torch.cat(
                [self.c_kv_cache[layer_idx], c_kv], dim=1
            )
            self.k_rope_cache[layer_idx] = torch.cat(
                [self.k_rope_cache[layer_idx], k_rope], dim=1
            )

        return self.c_kv_cache[layer_idx], self.k_rope_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the current sequence length stored in the cache.

        Args:
            layer_idx: Which layer's cache to check.

        Returns:
            The sequence length in the cache for the specified layer.
        """
        if len(self.c_kv_cache) <= layer_idx:
            return 0
        return self.c_kv_cache[layer_idx].shape[1]

    def get_max_length(self) -> Optional[int]:
        """Get the maximum length of the cache.

        Returns:
            None (dynamic cache has no maximum length).
        """
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder the cache for beam search.

        Args:
            beam_idx: Tensor of beam indices to select.
        """
        for layer_idx in range(len(self.c_kv_cache)):
            self.c_kv_cache[layer_idx] = self.c_kv_cache[layer_idx].index_select(0, beam_idx)
            self.k_rope_cache[layer_idx] = self.k_rope_cache[layer_idx].index_select(0, beam_idx)

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Convert to legacy cache format (tuple of tuples).

        Returns:
            Tuple of (c_kv, k_rope) tuples for each layer.
        """
        return tuple(
            (self.c_kv_cache[layer_idx], self.k_rope_cache[layer_idx])
            for layer_idx in range(len(self.c_kv_cache))
        )

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]
    ) -> Optional["MLADynamicCache"]:
        """Create an MLADynamicCache from legacy cache format.

        Args:
            past_key_values: Legacy cache format (tuple of (c_kv, k_rope) tuples).

        Returns:
            MLADynamicCache instance or None if input is None.
        """
        if past_key_values is None:
            return None

        cache = cls()
        for layer_idx, (c_kv, k_rope) in enumerate(past_key_values):
            cache.c_kv_cache.append(c_kv)
            cache.k_rope_cache.append(k_rope)

        if cache.c_kv_cache:
            cache._seen_tokens = cache.c_kv_cache[0].shape[1]

        return cache
