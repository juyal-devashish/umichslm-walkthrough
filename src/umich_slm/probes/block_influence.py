"""Block Influence probe (Men et al. 2024, ShortGPT).

For each layer/block, compute::

    BI = 1 - E[ cos(h_in, h_out) ]

over a held-out batch. Low BI means the block barely changed the residual
stream — a candidate for skipping or deduplication. We compute both per-layer
and per-block (4-layer block-aware) values; the per-block value is what the
I/O boundary heuristic acts on.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F


@torch.no_grad()
def block_influence(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    block_size: int = 4,
) -> Dict[str, Any]:
    """Run the block-influence probe.

    Args:
        model: A ``UMichSLMForCausalLM`` instance (already on the right device).
        input_ids: Tensor of shape ``[batch, seq_len]``.
        block_size: Layers per block (4 in the canonical 3:1 hybrid).

    Returns:
        Dict with keys:
          * ``block_influence_per_layer``: list[float] of length L.
          * ``block_influence_per_block``: list[float] of length L // block_size.
    """
    backbone = model.model

    # Run the embedding layer to get the residual stream entering layer 0.
    embeds = backbone.embed_tokens(input_ids)
    if backbone.global_tokens is not None and backbone.num_global_tokens > 0:
        # Globals are real residual-stream content from layer 0's perspective.
        embeds = backbone.global_tokens.prepend_to_embeddings(embeds)

    h = embeds
    per_layer: List[float] = []
    for layer in backbone.layers:
        h_in = h
        # Forward this layer in isolation. We pass attention_mask=None and let
        # the layer construct its causal mask itself; for SWA-globals layers
        # the mask builder reads ``num_protected_prefix`` from the layer.
        layer_out = layer(h_in)
        if isinstance(layer_out, tuple):
            h = layer_out[0]
        else:
            h = layer_out
        # Cosine over the last dimension, averaged over (B, T)
        cos = F.cosine_similarity(h_in, h, dim=-1)
        bi = 1.0 - cos.mean().item()
        per_layer.append(float(bi))

    n_blocks = len(per_layer) // block_size
    per_block = [
        float(sum(per_layer[i * block_size : (i + 1) * block_size]) / max(block_size, 1))
        for i in range(n_blocks)
    ]
    return {
        "block_influence_per_layer": per_layer,
        "block_influence_per_block": per_block,
    }
