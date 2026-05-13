"""Metric collectors for stacking-pilot training runs.

``collect_step_metrics(model, output)`` is called once per logged step inside
the training loop. It returns a flat dict that gets merged into the per-step
wandb payload and the metrics.jsonl line.

The collectors here are read-only: they walk the live model's modules to pull
out the running stats they need (per-block routing fraction, gate distribution
moments, etc.). They do not perform extra forward passes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..modules.global_tokens import GlobalTokens
from ..modules.mixture_of_depths import MoDOutput
from ..modules.transformer_block import UMichSLMMoDDecoderLayer


def _routing_fraction_per_block(model: nn.Module) -> Dict[str, float]:
    """Return the average MoD selected-mask fraction per inserted block, plus the global mean.

    We pull the most recent ``selected_mask`` saved on each MoD layer's router
    output. The training loop must populate this; if not present, the layer is
    skipped silently. Block index is inferred from the layer index using
    ``block_size`` from the config's ``stage_block_spec`` (or default 4).
    """
    cfg = getattr(model, "config", None)
    block_size = 4
    if cfg is not None and getattr(cfg, "stage_block_spec", None):
        first = cfg.stage_block_spec[0]
        block_size = len(first.get("block_layout", []))
    out: Dict[str, float] = {}
    fractions: list[float] = []
    layers = getattr(getattr(model, "model", model), "layers", [])
    for layer_idx, layer in enumerate(layers):
        if not isinstance(layer, UMichSLMMoDDecoderLayer):
            continue
        if not getattr(layer, "_is_inserted", False):
            continue
        last_mask = getattr(layer, "_last_selected_mask", None)
        if last_mask is None:
            continue
        block_idx = layer_idx // block_size
        frac = float(last_mask.float().mean().item())
        out[f"mod/routing_fraction/block_{block_idx}"] = frac
        fractions.append(frac)
    if fractions:
        out["mod/routing_fraction/mean"] = sum(fractions) / len(fractions)
        out["mod/routing_fraction/min"] = min(fractions)
        out["mod/routing_fraction/max"] = max(fractions)
    return out


def _global_token_utilization(model: nn.Module) -> Dict[str, float]:
    """Pairwise cosine similarity of GlobalTokens (collapse detector for T5b).

    A high pairwise similarity (e.g. >0.95) signals globals collapse.
    """
    backbone = getattr(model, "model", model)
    g: Optional[GlobalTokens] = getattr(backbone, "global_tokens", None)
    if g is None or g.num_tokens == 0:
        return {}
    with torch.no_grad():
        t = g.tokens.detach()
        if t.dim() != 2 or t.size(0) < 2:
            return {}
        normed = torch.nn.functional.normalize(t, dim=-1)
        # NxN cosine similarity matrix; ignore diagonal.
        sim = normed @ normed.T
        n = sim.size(0)
        eye = torch.eye(n, dtype=torch.bool, device=sim.device)
        offdiag = sim.masked_select(~eye)
        return {
            "swa/globals/pairwise_cos_mean": float(offdiag.mean().item()),
            "swa/globals/pairwise_cos_max": float(offdiag.max().item()),
            "swa/globals/norm_mean": float(t.norm(dim=-1).mean().item()),
        }


def _mtp_per_head_loss(output) -> Dict[str, float]:
    """Pull per-head Gloeckle MTP losses from the model output, if present."""
    out: Dict[str, float] = {}
    losses = getattr(output, "mtp_losses_per_head", None)
    if losses is None:
        # Older output type may store this on a sub-attribute
        losses = getattr(getattr(output, "mtp_output", None), "losses_per_depth", None)
    if losses is None:
        return out
    for i, l in enumerate(losses):
        if l is not None:
            out[f"mtp/head_{i}/loss"] = float(l)
    return out


def _deltanet_gate_stats(model: nn.Module) -> Dict[str, float]:
    """Mean / min / max of DeltaNet gates, if present.

    Walks named_modules looking for tensor attributes named ``last_gate_mean``
    that the DeltaNet layer (FLA-backed) exposes. If the wrapped FLA module
    doesn't expose this, the collector returns nothing — that's fine, it just
    means DeltaNet gate stats aren't logged for this run.
    """
    out: Dict[str, float] = {}
    means: list[float] = []
    for name, mod in model.named_modules():
        last = getattr(mod, "_last_gate_mean", None)
        if last is None:
            continue
        means.append(float(last))
    if means:
        out["deltanet/gate/mean"] = sum(means) / len(means)
        out["deltanet/gate/min"] = min(means)
        out["deltanet/gate/max"] = max(means)
    return out


def collect_step_metrics(model: nn.Module, output: Any) -> Dict[str, float]:
    """Return a flat metrics dict for the current step."""
    metrics: Dict[str, float] = {}
    metrics.update(_routing_fraction_per_block(model))
    metrics.update(_global_token_utilization(model))
    metrics.update(_mtp_per_head_loss(output))
    metrics.update(_deltanet_gate_stats(model))
    return metrics


def collect_post_transition_metrics(
    pre_transition_loss: float,
    post_transition_loss: float,
    *,
    stage_index: int,
) -> Dict[str, float]:
    """Metrics emitted after a stage transition (T3 detector)."""
    return {
        f"transition/stage_{stage_index}/pre_loss": float(pre_transition_loss),
        f"transition/stage_{stage_index}/post_loss": float(post_transition_loss),
        f"transition/stage_{stage_index}/loss_jump": float(
            post_transition_loss - pre_transition_loss
        ),
    }
