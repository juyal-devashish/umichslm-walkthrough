"""Frequency-scaled weight decay for the nested-width FFN.

The problem, measured. Row r of the FFN weights belongs to tier k, and that
block only receives gradient from tokens routed to tier >= k. Under the default
skewed prior the four blocks are trained by 100% / 40% / 18% / 6% of tokens, so
the WIDEST block -- which is also the LARGEST, 1536 of 3072 columns -- trains on
16.7x fewer tokens than the narrowest.

Decoupled weight decay ignores this completely: `w -= lr * wd * w` runs on every
parameter every step whether or not it was used. Over an ar10m run (26,037
steps, lr 1.75e-3, wd 0.1) the shrink factor is exp(-lr*wd*steps) ~ 0.01, and
tier 3 has 16.7x fewer gradient updates to push back with. Under the collapse we
actually measured it was far worse -- tier 3 saw 0.02% of tokens -- so those
columns decayed toward zero with essentially nothing resisting. That plausibly
makes the collapse SELF-REINFORCING: the wide columns rot, which makes them
genuinely useless, which confirms the router's decision to avoid them.

Fix: scale each row's decay by how often that row is actually trained, which
equalises decay-per-update across tiers. Rows used 6% of the time get 6% of the
decay.

Why not optimizer param groups: the tiers are SLICES of single tensors
(gate_proj.weight is [intermediate, hidden] and tier k is a row range), and a
tensor cannot be split across param groups. So the decay is applied here, and
the tensors are excluded from the optimizer's own weight_decay.

Usage:
    exempt = mat_ffn_wd_exempt_ids(model)
    groups = [{"params": [p for p in model.parameters() if id(p) not in exempt],
               "weight_decay": wd},
              {"params": [p for p in model.parameters() if id(p) in exempt],
               "weight_decay": 0.0}]
    ...
    optimizer.step()
    apply_frequency_scaled_decay(model, lr=current_lr, weight_decay=wd)
"""

from __future__ import annotations

from typing import List, Set, Tuple

import torch


def _routed_mat_mlps(model) -> list:
    from ..modules.swiglu import RoutedMatFormerSwiGLUMLP

    return [m for m in model.modules() if isinstance(m, RoutedMatFormerSwiGLUMLP)]


def mat_ffn_decay_tensors(model) -> List[Tuple[torch.nn.Parameter, torch.Tensor, int]]:
    """(param, per-row usage scale, dim the scale indexes) for every routed FFN.

    gate_proj/up_proj are [intermediate, hidden] so the tier axis is dim 0;
    down_proj is [hidden, intermediate] so it is dim 1.
    """
    out = []
    for m in _routed_mat_mlps(model):
        scale = m.tier_row_scale()
        for proj, dim in ((m.gate_proj, 0), (m.up_proj, 0), (m.down_proj, 1)):
            w = proj.weight
            out.append((w, scale.to(device=w.device, dtype=torch.float32), dim))
    return out


def mat_ffn_wd_exempt_ids(model) -> Set[int]:
    """ids of params whose decay this module owns -- exclude from the optimizer."""
    return {id(p) for p, _, _ in mat_ffn_decay_tensors(model)}


@torch.no_grad()
def apply_frequency_scaled_decay(model, *, lr: float, weight_decay: float) -> int:
    """Decoupled AdamW decay with a per-row frequency scale. Returns #tensors hit.

    Call AFTER optimizer.step(), with the CURRENT scheduled lr -- decoupled decay
    in AdamW is proportional to lr, so a decayed schedule must decay the decay
    too, or the tail of training over-shrinks.
    """
    if weight_decay <= 0.0 or lr <= 0.0:
        return 0
    n = 0
    for w, scale, dim in mat_ffn_decay_tensors(model):
        s = scale if dim == 0 else scale
        shape = [1] * w.dim()
        shape[dim] = -1
        w.mul_(1.0 - lr * weight_decay * s.view(shape).to(w.dtype))
        n += 1
    return n


@torch.no_grad()
def apply_frequency_scaled_lr(model, *, exponent: float, snapshot=None) -> int:
    """Per-tier LEARNING RATE, applied as a post-step correction on the UPDATE.

    Two-call protocol, because Adam is scale-invariant in the gradient:

        snap = snapshot_mat_ffn(model)     # before optimizer.step()
        optimizer.step()
        apply_frequency_scaled_lr(model, exponent=e, snapshot=snap)

    Scaling the GRADIENT would do nothing -- Adam normalises by sqrt(v), so a
    uniformly scaled gradient produces the same update. The per-tier rate has to
    act on the realised update instead: w <- before + scale**e * (after - before).

    Direction is deliberately a free parameter because the two standard arguments
    disagree, and this is cheaper to settle by experiment than by assertion:

      exponent = +0.5  scale = sqrt(usage). LOWER rate for rarely-trained tiers.
                       The batch-size argument: tier 3 is trained by 6% of tokens,
                       an effectively 16.7x smaller batch, so its gradient is
                       ~4x noisier and wants a smaller step for stability.
      exponent = -0.5  scale = 1/sqrt(usage). HIGHER rate for rare tiers.
                       The catch-up argument: tier 3 gets 16.7x fewer useful
                       updates over the run, so it trains 16.7x less.
      exponent =  0.0  off (default), bit-identical to plain AdamW.

    Only meaningful once occupancy is actually held near the prior. Under the
    collapsed router tier 3 saw 0.02% of tokens, where no rate can help.
    """
    if exponent == 0.0 or not snapshot:
        return 0
    n = 0
    for (w, scale, dim), before in zip(mat_ffn_decay_tensors(model), snapshot):
        shape = [1] * w.dim()
        shape[dim] = -1
        mult = scale.pow(exponent).view(shape).to(w.dtype)
        w.copy_(before + (w - before) * mult)
        n += 1
    return n


@torch.no_grad()
def snapshot_mat_ffn(model):
    """Clone the routed FFN weights so the update can be rescaled after the step."""
    return [w.detach().clone() for w, _, _ in mat_ffn_decay_tensors(model)]
