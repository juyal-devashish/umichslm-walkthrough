"""Linearity probe (Saunshi et al. 2024).

For each layer, fit a simple linear map ``A`` such that ``A h_l ≈ h_{l+1} - h_l``
on a held-out batch and report the coefficient of determination r². High r²
means the layer's residual update is well-approximated by a linear map of its
input — a Saunshi-style indicator of "linear" / "shallow" computation.

We solve the least-squares problem in closed form using the normal equations
on flattened (B*T, d) tensors. Reported per layer and (separately) for the
local layers (delta or swa) vs the MLA / global-attention layers, since the
plan asks for them split.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def _layer_kind_for(model: torch.nn.Module, layer_idx: int) -> str:
    """Return ``"local"`` (DeltaNet / SWA / standard) or ``"mla"`` for a layer."""
    cfg = model.config
    pattern = getattr(cfg, "hybrid_attention_ratio", None)
    if not pattern:
        return "local"
    entry = pattern[layer_idx % len(pattern)]
    if entry == "mla":
        return "mla"
    return "local"


@torch.no_grad()
def linearity_r2(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    ridge: float = 1e-3,
) -> Dict[str, Any]:
    """Compute per-layer linearity r².

    Args:
        model: ``UMichSLMForCausalLM`` instance.
        input_ids: Tensor of shape ``[batch, seq_len]``.
        ridge: Tiny ridge penalty to keep the closed-form solution stable.

    Returns:
        Dict with keys:
          * ``linearity_r2_per_layer``: list[float] of length L.
          * ``linearity_r2_local``: list[float] (subset of layers — local kind).
          * ``linearity_r2_mla``: list[float] (subset of layers — MLA kind).
    """
    backbone = model.model
    embeds = backbone.embed_tokens(input_ids)
    if backbone.global_tokens is not None and backbone.num_global_tokens > 0:
        embeds = backbone.global_tokens.prepend_to_embeddings(embeds)

    h = embeds
    per_layer: List[float] = []
    local_r2: List[float] = []
    mla_r2: List[float] = []

    for i, layer in enumerate(backbone.layers):
        h_in = h
        layer_out = layer(h_in)
        h_out = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        delta = h_out - h_in  # what the layer added to the residual stream

        # Flatten to (N, d).
        x = h_in.reshape(-1, h_in.size(-1)).double()
        y = delta.reshape(-1, delta.size(-1)).double()

        # Closed-form ridge regression: A = (X^T X + λI)^-1 X^T Y.
        d = x.size(-1)
        xtx = x.T @ x + ridge * torch.eye(d, dtype=x.dtype, device=x.device)
        try:
            A = torch.linalg.solve(xtx, x.T @ y)
        except RuntimeError:
            # Fall back to lstsq if solve diverges
            A = torch.linalg.lstsq(xtx, x.T @ y).solution

        y_hat = x @ A
        ss_res = ((y - y_hat) ** 2).sum().item()
        ss_tot = ((y - y.mean(dim=0, keepdim=True)) ** 2).sum().item()
        r2 = 1.0 - (ss_res / max(ss_tot, 1e-12))
        per_layer.append(float(r2))

        kind = _layer_kind_for(model, i)
        if kind == "mla":
            mla_r2.append(float(r2))
        else:
            local_r2.append(float(r2))

        h = h_out

    return {
        "linearity_r2_per_layer": per_layer,
        "linearity_r2_local": local_r2,
        "linearity_r2_mla": mla_r2,
    }
