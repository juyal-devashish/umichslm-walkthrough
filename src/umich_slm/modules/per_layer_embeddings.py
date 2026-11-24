"""Per-Layer Embeddings (PLE), Gemma-3n/4 style.

In a standard transformer the token embedding is consumed once, at the input. PLE
gives **every decoder layer** its own small token-embedding table
(``vocab x ple_dim``), projected up to ``hidden_size`` and added to that layer's
input. Each layer thus receives a token-specific, layer-specific signal that did
not have to survive the trip up the residual stream.

The per-layer tables are large (``vocab x ple_dim x num_layers``) but are sparse
lookups — only the rows for the current tokens are touched, and the tables can be
streamed from cheap memory. They therefore inflate **total** parameter count
without inflating the **active** compute / fast-memory footprint, which is the
mechanism behind Gemma's "E2B = 5.1B total / ~2B effective".

This module is intentionally simple: one shared embedding table holding all layers'
per-layer vectors concatenated, plus one projection per layer. It is a no-op unless
``config.use_ple`` is set.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PerLayerEmbeddings(nn.Module):
    """Per-layer token embeddings projected into the residual stream.

    Args:
        vocab_size: Token vocabulary size.
        num_layers: Number of decoder layers (one per-layer table each).
        ple_dim: Width of each per-layer embedding (Gemma default 256).
        hidden_size: Model hidden size the per-layer embedding is projected to.
        init_std: Init std for the embedding table.
        proj_init_std: Init std for the projections (small, so PLE starts as a
            gentle perturbation rather than dominating the residual stream).
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        ple_dim: int,
        hidden_size: int,
        init_std: float = 0.02,
        proj_init_std: float = 0.02,
    ):
        super().__init__()
        if num_layers <= 0 or ple_dim <= 0:
            raise ValueError("num_layers and ple_dim must be > 0 for PLE.")
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.ple_dim = ple_dim
        self.hidden_size = hidden_size

        # One table holding all layers' per-layer vectors, concatenated along dim.
        self.embed = nn.Embedding(vocab_size, num_layers * ple_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=init_std)

        # One projection per layer: ple_dim -> hidden_size.
        self.projs = nn.ModuleList(
            [nn.Linear(ple_dim, hidden_size, bias=False) for _ in range(num_layers)]
        )
        for p in self.projs:
            nn.init.normal_(p.weight, mean=0.0, std=proj_init_std)
        # Per-layer RMS-style norm on the looked-up embedding keeps scales stable.
        self.norm = nn.RMSNorm(ple_dim) if hasattr(nn, "RMSNorm") else None

    def forward_layer(
        self, input_ids: torch.LongTensor, layer_idx: int, prefix_len: int = 0
    ) -> torch.Tensor:
        """Return the per-layer embedding contribution for ``layer_idx``.

        Args:
            input_ids: (batch, seq) content token ids.
            layer_idx: Which decoder layer (0-based).
            prefix_len: Number of non-content positions prepended to the hidden
                states (e.g. SWA global tokens / meta tokens). The returned tensor
                is zero-padded over the prefix so it aligns with the hidden states.

        Returns:
            (batch, prefix_len + seq, hidden_size) tensor to add to the layer input.
        """
        table = self.embed(input_ids)  # (B, S, num_layers * ple_dim)
        start = layer_idx * self.ple_dim
        vec = table[..., start : start + self.ple_dim]  # (B, S, ple_dim)
        if self.norm is not None:
            vec = self.norm(vec)
        out = self.projs[layer_idx](vec)  # (B, S, hidden)
        if prefix_len:
            pad = out.new_zeros(out.shape[0], prefix_len, out.shape[-1])
            out = torch.cat([pad, out], dim=1)
        return out

    def total_params(self) -> int:
        """Total parameters owned by PLE (dominated by the embedding tables)."""
        return sum(p.numel() for p in self.parameters())

    def table_params(self) -> int:
        """Just the (large, streamable) per-layer embedding tables."""
        return self.embed.weight.numel()
