"""Engram: Hash-Based Conditional Memory Module.

Engram uses O(1) hash lookups to retrieve pre-trained n-gram embeddings and merges
them into specific transformer layers via context-aware gating. This offloads static
pattern retrieval from the transformer backbone, freeing attention capacity for reasoning.

Reference: https://arxiv.org/abs/2601.07372

The module works in 3 steps per designated layer:
1. Hash: Map input token n-grams to indices via multi-head XOR hashing
2. Lookup: Retrieve learned embeddings from tables indexed by those hashes
3. Gate + Merge: Use a context-aware gate to blend retrieved embeddings into
   hidden states as a residual addition
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn

from ..configuration_umichslm import UMichSLMConfig


class EngramHash(nn.Module):
    """Deterministic multi-head n-gram hashing.

    Maps input token n-grams to hash indices using XOR-based hashing with
    layer-specific odd multipliers. Multiple hash heads per n-gram order
    provide collision mitigation.

    Args:
        ngram_sizes: List of n-gram orders (e.g., [2, 3] for bigrams + trigrams).
        num_heads: Number of hash heads per n-gram order.
        table_size: Size of the hash table (should be prime for good distribution).
        seed: Seed for deterministic multiplier generation.
    """

    def __init__(
        self,
        ngram_sizes: List[int],
        num_heads: int,
        table_size: int,
        seed: int = 42,
    ):
        super().__init__()
        self.ngram_sizes = ngram_sizes
        self.num_heads = num_heads
        self.table_size = table_size
        self.total_heads = num_heads * len(ngram_sizes)

        # Generate deterministic odd multipliers from seed
        rng = torch.Generator()
        rng.manual_seed(seed)
        # For each ngram_size * num_heads, we need ngram_size multipliers
        # Shape: (len(ngram_sizes), num_heads, max_ngram_size)
        max_ngram = max(ngram_sizes)
        multipliers = torch.randint(
            1, table_size, (len(ngram_sizes), num_heads, max_ngram),
            generator=rng,
        )
        # Make them odd for better hash distribution
        multipliers = multipliers | 1
        self.register_buffer("multipliers", multipliers)

    def forward(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Compute hash indices for input token n-grams.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Hash indices of shape (batch, seq_len, total_heads) where
            total_heads = num_heads * len(ngram_sizes).
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        all_hashes = []

        for ng_idx, ngram_size in enumerate(self.ngram_sizes):
            for head_idx in range(self.num_heads):
                # XOR-combine shifted token IDs with multipliers
                hash_val = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
                for offset in range(ngram_size):
                    # Shift input_ids by offset (tokens before position use padding=0)
                    if offset == 0:
                        shifted = input_ids
                    elif offset >= seq_len:
                        # Entire sequence is padding
                        shifted = torch.zeros_like(input_ids)
                    else:
                        # Pad with zeros at the beginning
                        padding = torch.zeros(
                            batch_size, offset, dtype=torch.long, device=device
                        )
                        shifted = torch.cat([padding, input_ids[:, :-offset]], dim=1)

                    mult = self.multipliers[ng_idx, head_idx, offset].item()
                    hash_val = hash_val ^ (shifted * mult)

                # Reduce mod table_size
                hash_val = hash_val % self.table_size
                all_hashes.append(hash_val)

        # Stack: (batch, seq_len, total_heads)
        return torch.stack(all_hashes, dim=-1)


class EngramMemory(nn.Module):
    """Embedding table lookup for Engram.

    Stores embedding tables indexed by hash values. Given hash indices,
    retrieves and projects embeddings to hidden_size.

    Args:
        num_tables: Total number of hash tables (num_heads * len(ngram_sizes)).
        table_size: Size of each hash table.
        embed_dim: Embedding dimension per table.
        hidden_size: Model hidden size (projection target).
    """

    def __init__(
        self,
        num_tables: int,
        table_size: int,
        embed_dim: int,
        hidden_size: int,
    ):
        super().__init__()
        self.num_tables = num_tables
        self.embed_dim = embed_dim

        # One embedding table per hash head
        self.tables = nn.ModuleList([
            nn.Embedding(table_size, embed_dim) for _ in range(num_tables)
        ])

        # Project concatenated embeddings to hidden_size
        self.proj = nn.Linear(num_tables * embed_dim, hidden_size, bias=False)

    def forward(self, hash_indices: torch.LongTensor) -> torch.Tensor:
        """Look up and project embeddings from hash indices.

        Args:
            hash_indices: Hash indices of shape (batch, seq_len, num_tables).

        Returns:
            Retrieved embeddings of shape (batch, seq_len, hidden_size).
        """
        # Gather embeddings from each table
        embeds = []
        for i, table in enumerate(self.tables):
            # hash_indices[:, :, i] -> (batch, seq_len)
            embeds.append(table(hash_indices[:, :, i]))

        # Concatenate: (batch, seq_len, num_tables * embed_dim)
        concat = torch.cat(embeds, dim=-1)

        # Project to hidden_size
        return self.proj(concat)


class EngramGate(nn.Module):
    """Context-aware gating for Engram.

    Computes a gate value based on the normalized dot product between
    hidden states and retrieved embeddings, scaled and passed through sigmoid.

    Gate = sigmoid(sqrt(abs(dot(RMSNorm(Q), RMSNorm(M)) / sqrt(d))) * sign(...))

    Args:
        hidden_size: Model hidden size.
        eps: Epsilon for RMSNorm numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        # Learnable RMSNorm weights for Q and M
        self.q_weight = nn.Parameter(torch.ones(hidden_size))
        self.m_weight = nn.Parameter(torch.ones(hidden_size))

    def _rmsnorm(self, x: torch.Tensor, weight: nn.Parameter) -> torch.Tensor:
        """Apply RMSNorm with given weight."""
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (weight * x).to(input_dtype)

    def forward(
        self, hidden_states: torch.Tensor, retrieved: torch.Tensor
    ) -> torch.Tensor:
        """Compute gated output.

        Args:
            hidden_states: Hidden states (Q) of shape (batch, seq_len, hidden_size).
            retrieved: Retrieved embeddings (M) of shape (batch, seq_len, hidden_size).

        Returns:
            Gated output of shape (batch, seq_len, hidden_size), ready for
            residual addition.
        """
        # Normalize Q and M
        q_norm = self._rmsnorm(hidden_states, self.q_weight)
        m_norm = self._rmsnorm(retrieved, self.m_weight)

        # Dot product per position: (batch, seq_len)
        dot = (q_norm * m_norm).sum(dim=-1, keepdim=True)

        # Scale by sqrt(d)
        d = self.hidden_size
        scaled = dot / math.sqrt(d)

        # Gate = sigmoid(sqrt(abs(scaled)) * sign(scaled))
        gate = torch.sigmoid(torch.sqrt(torch.abs(scaled) + self.eps) * torch.sign(scaled))

        # Apply gate to retrieved embeddings
        return gate * retrieved


class EngramModule(nn.Module):
    """Top-level Engram module combining hash, memory, and gate.

    Applied after designated transformer layers as a residual addition:
        hidden_states = hidden_states + engram(hidden_states, input_ids)

    Supports generation by maintaining a token ID buffer that accumulates
    across generation steps.

    Args:
        config: Model configuration.
        layer_idx: Index of the transformer layer this module is attached to.
    """

    def __init__(self, config: UMichSLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        ngram_sizes = config.engram_ngram_sizes or [2, 3]
        num_heads = config.engram_num_heads
        embed_dim = config.engram_embed_dim
        table_size = config.engram_table_size
        seed = config.engram_seed

        # Use a different seed per layer for diverse hash functions
        layer_seed = seed + layer_idx * 1000

        self.hash_fn = EngramHash(
            ngram_sizes=ngram_sizes,
            num_heads=num_heads,
            table_size=table_size,
            seed=layer_seed,
        )

        total_tables = num_heads * len(ngram_sizes)
        self.memory = EngramMemory(
            num_tables=total_tables,
            table_size=table_size,
            embed_dim=embed_dim,
            hidden_size=config.hidden_size,
        )

        self.gate = EngramGate(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Optional short convolution for local context
        self.use_conv = getattr(config, "engram_use_conv", False)
        if self.use_conv:
            conv_kernel = getattr(config, "engram_conv_kernel_size", 3)
            self.conv = nn.Conv1d(
                config.hidden_size,
                config.hidden_size,
                kernel_size=conv_kernel,
                padding=conv_kernel // 2,
                groups=config.hidden_size,
            )

        # Token buffer for generation (not a parameter, not saved)
        self._token_buffer: Optional[torch.LongTensor] = None

    def reset_token_buffer(self):
        """Reset the token buffer (call at start of new sequence)."""
        self._token_buffer = None

    def update_token_buffer(self, input_ids: torch.LongTensor):
        """Update the token buffer with new tokens.

        During generation, tokens arrive one at a time. This method accumulates
        them so the hash function always has enough context for n-gram computation.

        Args:
            input_ids: New token IDs of shape (batch, new_seq_len).
        """
        if self._token_buffer is None:
            self._token_buffer = input_ids
        else:
            self._token_buffer = torch.cat([self._token_buffer, input_ids], dim=1)

    def get_token_ids(
        self, input_ids: Optional[torch.LongTensor], past_key_values
    ) -> torch.LongTensor:
        """Get full token ID history for hashing.

        During prefill, returns input_ids directly. During generation, uses
        the accumulated token buffer.

        Args:
            input_ids: Current input token IDs.
            past_key_values: Cache (used to detect generation mode).

        Returns:
            Full token ID history for hash computation.
        """
        if past_key_values is not None and self._token_buffer is not None:
            return self._token_buffer
        return input_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute engram residual.

        Args:
            hidden_states: Hidden states of shape (batch, seq_len, hidden_size).
            input_ids: Token IDs of shape (batch, token_seq_len). During generation,
                this should be the full token history.

        Returns:
            Engram output of shape (batch, seq_len, hidden_size), ready for
            residual addition to hidden_states.
        """
        # Hash input tokens
        hash_indices = self.hash_fn(input_ids)

        # During generation, we may have more tokens in buffer than hidden states
        # Only take the last seq_len hash positions to match hidden_states
        seq_len = hidden_states.shape[1]
        if hash_indices.shape[1] > seq_len:
            hash_indices = hash_indices[:, -seq_len:, :]

        # Lookup embeddings
        retrieved = self.memory(hash_indices)

        # Optional convolution for local context
        if self.use_conv:
            # Conv1d expects (batch, channels, seq_len)
            retrieved = self.conv(retrieved.transpose(1, 2)).transpose(1, 2)

        # Gate and return
        return self.gate(hidden_states, retrieved)
