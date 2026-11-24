"""Bench-only: pin dynamic-compute routers to their CONVERGED allocation.

Why this exists. Every routed arm here is benched at random init, and the
"menu-symmetry" rule (docs/arch_naming.md) assumes a random-init router is
~uniform over its menu, so realized compute equals the menu mean. That rule
holds for the router *in isolation* -- measured on N(0,1) input a 4-lane
PathwayRouter gives fractions [0.25, 0.27, 0.28, 0.20] -- but NOT for a router
sitting at layer 10 of an untrained network, whose hidden states carry enough
systematic structure to skew the argmax. Measured on `think_suffix` at init:
lane fractions [0.0, 0.0, 0.451, 0.549], i.e. a realized 3.10 blocks/token
against a 1.75 budget. Benching that arm naively overstates its steady-state
cost by ~77%.

The same problem, in the other direction, affects any arm whose budget differs
from its menu mean -- `swa_scribe_mopr_b1` (budget 1.0 on a menu whose mean is
2.0) benches as if it were plain mopr and shows none of its saving.

What this does. Replace the router's argmax with a deterministic assignment
whose per-tier occupancy matches the budget, then bench that. It changes only
WHICH tokens go where, never how much work each tier does, so the resulting
ms/step is the steady-state number a trained cell would pay.

Two details that matter for honesty:
  * fractions are the closest-to-uniform vector satisfying the budget, not a
    two-point mix -- tier occupancy drives gather widths, and collapsing onto
    two tiers would flatter the slice/gather path;
  * assignment is SCATTERED (strided by a coprime step), not contiguous. A
    contiguous assignment would hand the permute/slice dispatch a best case it
    never sees in training.

Usage (bench only -- never import this from a training path):

    from umich_slm.utils.forced_routing import force_realized_routing
    force_realized_routing(model)     # reads each module's own menu + budget
"""

from typing import List, Optional, Sequence

import torch


def budget_fractions(depths: Sequence[float], budget: float) -> List[float]:
    """Closest-to-uniform tier fractions with sum(f_k * depths_k) == budget.

    Start uniform, then move mass along the depth axis toward the budget by
    water-filling from the extreme tiers inward. Falls back to a bracketing
    two-point mix when the budget cannot be met any other way (i.e. it sits
    outside the achievable mean range, which the config validator forbids).
    """
    d = [float(x) for x in depths]
    k = len(d)
    if k == 0:
        return []
    lo, hi = min(d), max(d)
    b = min(max(float(budget), lo), hi)
    f = [1.0 / k] * k
    cur = sum(fi * di for fi, di in zip(f, d))
    if abs(cur - b) < 1e-9:
        return f
    # Move mass between the lightest and heaviest tiers until the mean hits b.
    i_lo, i_hi = d.index(lo), d.index(hi)
    if cur > b:
        src, dst = i_hi, i_lo  # too expensive: shift mass down
    else:
        src, dst = i_lo, i_hi  # too cheap: shift mass up
    span = d[dst] - d[src]
    if abs(span) < 1e-12:
        return f
    shift = (b - cur) / span
    shift = max(-f[src], min(f[src], shift)) if shift < 0 else min(f[src], shift)
    f[src] -= abs(shift) if src != dst else 0.0
    f[dst] += abs(shift)
    tot = sum(f)
    return [x / tot for x in f]


def _scattered_assignment(n: int, fracs: Sequence[float], device) -> torch.Tensor:
    """Deterministic, SCATTERED tier ids with the requested occupancy.

    Strided by a step coprime with n so tiers interleave across the sequence.
    Contiguous runs would give the permute/slice dispatch an unrealistically
    friendly layout.
    """
    counts = [int(round(f * n)) for f in fracs]
    counts[-1] = n - sum(counts[:-1])
    ids = torch.empty(n, dtype=torch.long, device=device)
    order = torch.arange(n, device=device)
    step = 9973 % max(1, n)  # large prime, wraps deterministically
    while step > 1 and n % step == 0:
        step -= 1
    perm = (order * max(1, step)) % n if step > 1 else order
    pos = 0
    for tier, c in enumerate(counts):
        if c <= 0:
            continue
        ids[perm[pos : pos + c]] = tier
        pos += c
    return ids


def force_realized_routing(model, verbose: bool = True) -> int:
    """Pin every routed module in ``model`` to its budget-consistent allocation.

    Returns the number of modules pinned. Safe to call on a model with no
    routed modules (returns 0).
    """
    n = 0
    for mod in model.modules():
        # W1 mat_ffn: nested-width router
        if hasattr(mod, "widths") and hasattr(mod, "_forced_tier") and hasattr(mod, "budget"):
            # A SKEWED arm must bench at its skew, not at the closest-to-uniform
            # vector that happens to share its mean -- tier occupancy drives the
            # gather widths, and the two differ a lot once the prior is skewed
            # (0.60/0.22/0.12/0.06 against 0.25 each). `tier_prior()` is the
            # module's own definition, so the bench and the training path cannot
            # disagree. It reports the CURRENT stage, which is the target prior
            # for any module not mid-warmup.
            fr = (
                mod.tier_prior()
                if hasattr(mod, "tier_prior")
                else budget_fractions(mod.widths, mod.budget)
            )
            mod._forced_tier_fracs = fr
            mod._forced_tier = None  # fracs take precedence
            n += 1
            if verbose:
                print(
                    f"  [forced] mat_ffn widths={mod.widths} budget={mod.budget:.1f} "
                    f"-> fracs {[round(x, 3) for x in fr]}"
                )
            continue
        # MoP / Pathways: lane-depth router
        if hasattr(mod, "depths") and hasattr(mod, "router") and hasattr(mod, "budget"):
            fr = budget_fractions(mod.depths, mod.budget)
            mod._forced_lane_fracs = fr
            # Disable the deep-lane starvation floor. It is a WARMUP mechanism
            # (first pathway_floor_steps steps only), so leaving it armed makes
            # the bench report the warmup allocation rather than steady state --
            # measured: it pushed think_suffix to 1.99 blocks/token against a
            # 1.75 budget even with the lane assignment already pinned.
            if getattr(mod, "floor_frac", 0.0):
                mod.floor_frac = 0.0
            n += 1
            if verbose:
                print(
                    f"  [forced] pathways depths={list(mod.depths)} budget={mod.budget} "
                    f"-> fracs {[round(x, 3) for x in fr]}"
                )
    return n


def apply_forced_lane(mod, x: torch.Tensor) -> Optional[torch.Tensor]:
    """Return a forced [B,T] lane assignment for ``mod``, or None if not pinned."""
    fr = getattr(mod, "_forced_lane_fracs", None)
    if not fr:
        return None
    b, t = x.shape[0], x.shape[1]
    flat = _scattered_assignment(b * t, fr, x.device)
    return flat.view(b, t)


def apply_forced_tier(mod, n: int, device) -> Optional[torch.Tensor]:
    """Return a forced [N] tier assignment for ``mod``, or None if not pinned."""
    fr = getattr(mod, "_forced_tier_fracs", None)
    if not fr:
        return None
    return _scattered_assignment(n, fr, device)
