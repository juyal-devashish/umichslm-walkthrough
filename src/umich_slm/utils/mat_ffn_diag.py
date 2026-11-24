"""Is a routed nested FFN routing on CONTENT or on POSITION?

Occupancy cannot answer that. `mat_ffn_thresh` held mean width 767.9/768 for a
whole run while its thresholds sat at 0 and `_causal_budget_clip` handed the
wide tiers to the earliest positions of every sequence (2026-09-18: granted
width by position decile 2460/1382/768/768/384/.../384, identical in every
layer). A positional schedule satisfies any occupancy target exactly.

`routing_report` measures the two things that do distinguish them:

  * position profile -- granted width per position decile and its Spearman
    correlation with position. A content router is flat in expectation.
  * width benefit -- per-token NLL with every routed layer forced to the
    NARROWEST tier minus NLL forced to the WIDEST. That is how much this token
    gains from width. A router that has learned anything grants wide tiers to
    the tokens with the larger gain, so Spearman(granted width, gain) > 0 and
    the widest-granted quintile gains more than the rest.

Forward-only; runs on CPU for small models.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List

import torch
import torch.nn.functional as F

from ..modules.swiglu import RoutedMatFormerSwiGLUMLP


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = (ra.norm() * rb.norm()).item()
    return float((ra * rb).sum().item() / den) if den > 0 else 0.0


@contextmanager
def _record_clip(mods: List[RoutedMatFormerSwiGLUMLP]):
    """Capture each layer's (wished, granted) tier assignment, shape [B, S]."""
    rec: Dict[int, list] = {i: [] for i in range(len(mods))}
    saved = []
    for i, m in enumerate(mods):
        orig = m._causal_budget_clip

        def wrapped(choice, shape, _orig=orig, _i=i):
            out = _orig(choice, shape)
            b, s = int(shape[0]), int(shape[1])
            rec[_i].append((choice.detach().view(b, s).cpu(), out.detach().view(b, s).cpu()))
            return out

        m._causal_budget_clip = wrapped
        saved.append((m, orig))
    try:
        yield rec
    finally:
        for m, orig in saved:
            m._causal_budget_clip = orig


def _token_nll(model, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=input_ids).logits.float()
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1), reduction="none"
    ).view(input_ids.shape[0], -1)


@torch.no_grad()
def routing_report(model, input_ids: torch.Tensor) -> dict:
    """Routing diagnostics for every `RoutedMatFormerSwiGLUMLP` in `model`.

    `input_ids` is [B, S] on the model's device. The model is put in eval mode.
    """
    mods = [m for m in model.modules() if isinstance(m, RoutedMatFormerSwiGLUMLP)]
    if not mods:
        return {}
    was_training = model.training
    model.eval()
    b, s = input_ids.shape
    widths = torch.tensor(mods[0].widths, dtype=torch.float32)

    # Router logits, for the marginal-vs-nested pass rates below. The hook must
    # return None: a forward hook's return value REPLACES the module output, and
    # an earlier lambda returning `dict.setdefault(...)` swapped the router
    # output for a CPU copy (device mismatch on GPU, silent on CPU).
    logits_rec: Dict[int, torch.Tensor] = {}

    def _keep(k):
        def hook(_m, _i, out):
            logits_rec.setdefault(k, out.detach().float().cpu())
            return None

        return hook

    hooks = [m.router.register_forward_hook(_keep(i)) for i, m in enumerate(mods)]
    try:
        with _record_clip(mods) as rec:
            _token_nll(model, input_ids)
    finally:
        for h in hooks:
            h.remove()
    wished = torch.stack([widths[rec[i][0][0]] for i in range(len(mods))])  # [L, B, S]
    granted = torch.stack([widths[rec[i][0][1]] for i in range(len(mods))])
    g_mean = granted.mean(0)  # [B, S], mean over routed layers

    # `_forced_tier_fracs` (the random-routing control) takes precedence over
    # `_forced_tier` in forward, so park it for the counterfactual passes.
    parked = [getattr(m, "_forced_tier_fracs", None) for m in mods]
    for m in mods:
        m._forced_tier_fracs = None
        m._forced_tier = 0
    nll_narrow = _token_nll(model, input_ids)
    for m in mods:
        m._forced_tier = len(m.widths) - 1
    nll_wide = _token_nll(model, input_ids)
    for m, p in zip(mods, parked):
        m._forced_tier = None
        m._forced_tier_fracs = p
    gain = (nll_narrow - nll_wide).cpu()  # [B, S-1]; position t predicts t+1
    g_pred = g_mean[:, :-1]

    pos = torch.arange(s, dtype=torch.float32).expand(b, s)
    deciles = [
        round(float(g_mean[:, i * s // 10 : (i + 1) * s // 10].mean()), 1) for i in range(10)
    ]
    # Per layer, per tier k >= 1: the share of tokens clearing theta_k on its
    # own (MARGINAL) vs the share actually reaching tier >= k, which needs every
    # threshold <= k (NESTED, the pre-clip wish), vs the target. Marginal on
    # target with nested below it is the under-fill "quantile_nested" fixes.
    pass_rates = []
    for i, m in enumerate(mods):
        if not hasattr(m, "route_thresh") or i not in logits_rec:
            continue
        tail = torch.softmax(logits_rec[i], dim=-1).flip(-1).cumsum(-1).flip(-1)[:, 1:]
        th = m.route_thresh.detach().float().cpu().view(1, -1)
        ch = rec[i][0][0].flatten()
        fr = m.tier_fracs if m.tier_fracs else m.tier_prior()
        pass_rates.append({
            "marginal": [round(float(x), 4) for x in (tail > th).float().mean(0)],
            "nested": [round(float((ch >= k).float().mean()), 4) for k in range(1, len(fr))],
            "target": [round(float(sum(fr[k:])), 4) for k in range(1, len(fr))],
        })
    q = torch.quantile(g_pred.flatten(), 0.8)
    top = g_pred >= q
    model.train(was_training)
    return {
        "n_layers": len(mods),
        "seq_len": s,
        "theta": [[float(f"{float(x):.3g}") for x in m.route_thresh] for m in mods
                  if hasattr(m, "route_thresh")],
        "tier_pass_rates": pass_rates,
        "mean_wished": round(float(wished.mean()), 1),
        "mean_granted": round(float(granted.mean()), 1),
        "granted_by_position_decile": deciles,
        "spearman_width_position": round(_spearman(g_mean, pos), 4),
        "mean_gain_nats": round(float(gain.mean()), 4),
        "spearman_width_gain": round(_spearman(g_pred, gain), 4),
        "spearman_position_gain": round(_spearman(pos[:, :-1], gain), 4),
        # widest-granted fifth vs the rest: a content router gives the first
        # the larger gain. (A ratio misleads when the mean gain is ~0.)
        "gain_top_quintile_nats": round(float(gain[top].mean()), 4) if bool(top.any()) else None,
        "gain_rest_nats": round(float(gain[~top].mean()), 4) if bool((~top).any()) else None,
    }
