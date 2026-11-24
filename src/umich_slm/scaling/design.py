"""Target-region metrics, design diagnostics, and the cost-aware acquisition.

Three jobs, all of them answers to the same complaint: a scaling-law fit was
being judged by how well it reproduced the cells it was fitted on, which on this
project is a number below the seed noise floor and therefore nearly information
-free about the thing the fit is for.

1. ``target_metrics`` / ``TargetRegion`` -- score a fit where the decision lives
   (the deep80 ~1B model at TPR 350/700/1400), not in sample.

2. ``design_diagnostic`` -- the cheap/expensive-half check from Li et al. (2026),
   arXiv 2604.22753, table 4. On our grid "fit on everything" is NOT the best
   design: for ``qwen3_like`` against the held-out TPR ladder it scores target
   R^2 = -1.61 while the five CHEAPEST cells score +0.78 at 14% of the FLOPs.
   Any all-data fit should be reported next to this diagnostic, so that a design
   that loses to half its own pool cannot be quoted as if it were the answer.

3. ``score_candidates`` / ``plan_batch`` -- the SL^2 acquisition: expected
   reduction in target-region MSPE per unit cost, used to pick which cells to buy
   next. This replaces an earlier ad-hoc design study that scored candidates by
   the bootstrap SCATTER of the fitted parameters. Scatter is variance only, so
   that study was structurally blind to bias -- and the dominant error in our
   1B prediction is bias (one-signed, growing: -0.018/-0.045/-0.058 nats at TPR
   700/1400/2800).

Two departures from the paper, both because it says noise handling is its own
weakest axis and we have a measured sigma:

* ``sigma`` is a real measurement here (0.0127 nats), so it is what enters the
  Gauss-Newton curvature rather than a nuisance to be tuned.
* The candidate pool may contain REPEAT SEEDS of configs already observed. The
  paper's pool holds each config once and deletes it on selection, so it can
  never choose to re-run a config to average noise down. When sigma is
  comparable to the effect you are chasing -- ours is, the cross-seed bar is
  0.036 nats and arm gaps are 0.01-0.04 -- a second seed is sometimes the best
  thing the budget can buy, and the acquisition should be allowed to say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .basins import DEFAULT_SIGMA, Basin, cluster_basins, local_covariance, mspe, multistart
from .laws import Law

__all__ = [
    "TargetRegion",
    "DEEP80_N_ACTIVE",
    "Candidate",
    "CandidateScore",
    "target_metrics",
    "design_diagnostic",
    "score_candidates",
    "plan_batch",
]

# deep80 ~1B active params -- the model this whole grid exists to price.
DEEP80_N_ACTIVE = 1.018e9
# One-dimensional quadrature grid size for the inter-basin expectation; the
# paper uses Q = 500.
_QUAD = 500
# numpy >= 2 renamed trapz to trapezoid; bind once rather than branching per call.
_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass(frozen=True)
class TargetRegion:
    """The configurations a fit is judged on. ``points`` is ``(n, 2)`` of (N, D)."""

    points: np.ndarray
    labels: Tuple[str, ...]

    @classmethod
    def deep80(
        cls, tprs: Sequence[int] = (350, 700, 1400), n_active: float = DEEP80_N_ACTIVE
    ) -> "TargetRegion":
        pts = np.array([[n_active, t * n_active] for t in tprs], dtype=float)
        return cls(points=pts, labels=tuple(f"1B/TPR{t}" for t in tprs))

    @classmethod
    def from_cells(cls, cells: Sequence) -> "TargetRegion":
        """Target region defined by held-out CELLS, so it can be scored."""
        pts = np.array([[c.N, c.D] for c in cells], dtype=float)
        return cls(points=pts, labels=tuple(f"{c.size}/TPR{c.tpr}" for c in cells))

    def __len__(self) -> int:
        return len(self.points)

    @property
    def N(self) -> np.ndarray:
        return self.points[:, 0]

    @property
    def D(self) -> np.ndarray:
        return self.points[:, 1]


# ------------------------------------------------------------ target metrics
def target_metrics(
    law: Law,
    theta: np.ndarray,
    test: Sequence,
    *,
    sigma: float = DEFAULT_SIGMA,
) -> Dict[str, float]:
    """Error of a fit on held-out cells, in nats.

    ``rmse`` and ``bias`` are the headline numbers because they are in the unit
    our decisions are in; ``r2`` is reported too but is unstable when the target
    cells span little loss (our TPR ladder spans 0.066 nats), so it is secondary.
    """
    N = np.array([c.N for c in test], dtype=float)
    D = np.array([c.D for c in test], dtype=float)
    y = np.array([c.L for c in test], dtype=float)
    e = law.predict(theta, N, D) - y
    ss = float(np.sum((y - y.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "bias": float(np.mean(e)),
        "max_abs": float(np.max(np.abs(e))),
        "r2": (1.0 - float(np.sum(e**2)) / ss) if ss > 0 else float("nan"),
        "n_test": len(test),
        "at_noise": bool(np.sqrt(np.mean(e**2)) <= sigma),
    }


# --------------------------------------------------------- design diagnostic
def _flops(cell) -> float:
    return 6.0 * cell.N * cell.D


def design_diagnostic(
    law: Law,
    pool: Sequence,
    test: Sequence,
    *,
    sigma: float = DEFAULT_SIGMA,
    n_starts: int = 256,
    seed: int = 0,
    extra: Optional[Dict[str, Sequence]] = None,
) -> List[Dict[str, object]]:
    """Fit several designs drawn from ``pool`` and score each on ``test``.

    Returns rows sorted by target RMSE ascending, so a design that beats
    "all data" appears above it and cannot be overlooked.
    """
    order = sorted(pool, key=_flops)
    half = len(order) // 2
    designs: Dict[str, Sequence] = {
        "all data": list(pool),
        "cheap half": order[:half],
        "expensive half": order[half:],
    }
    if extra:
        designs.update(extra)

    rows: List[Dict[str, object]] = []
    for name, design in designs.items():
        if len(design) < law.n_params:
            rows.append(
                {
                    "design": name,
                    "n": len(design),
                    "flops": sum(map(_flops, design)),
                    "error": f"needs >= {law.n_params} cells",
                }
            )
            continue
        N = np.array([c.N for c in design], dtype=float)
        D = np.array([c.D for c in design], dtype=float)
        L = np.array([c.L for c in design], dtype=float)
        opt = multistart(law, N, D, L, n_starts=n_starts, seed=seed)
        if not opt:
            rows.append({"design": name, "n": len(design), "error": "no fit"})
            continue
        row: Dict[str, object] = {
            "design": name,
            "n": len(design),
            "flops": sum(map(_flops, design)),
            "in_sample_rmse": opt[0].rmse,
        }
        row.update(target_metrics(law, opt[0].theta, test, sigma=sigma))
        row.update(law.report(opt[0].theta, DEEP80_N_ACTIVE))
        rows.append(row)
    rows.sort(key=lambda r: r.get("rmse", float("inf")))
    return rows


# ------------------------------------------------------------- acquisition
@dataclass(frozen=True)
class Candidate:
    """A runnable cell. ``repeat`` marks a re-run of an already-observed config."""

    arm: str
    size: str
    tpr: int
    N: float
    D: float
    cost: float
    repeat: bool = False

    @property
    def key(self) -> Tuple[str, str, int]:
        return (self.arm, self.size, self.tpr)

    def label(self) -> str:
        return f"{self.size}/TPR{self.tpr}" + (" (repeat seed)" if self.repeat else "")


@dataclass
class CandidateScore:
    candidate: Candidate
    d_intra: float
    d_inter: float
    cost: float
    score: float

    @property
    def total(self) -> float:
        return self.d_intra + self.d_inter


def _basin_predictive(law: Law, basin: Basin, N: float, D: float, sigma: float):
    """Predictive mean and variance of one basin at a single candidate point."""
    Jx = np.atleast_2d(law.jacobian(basin.theta, np.array([N]), np.array([D])))
    mu = float(law.predict(basin.theta, np.array([N]), np.array([D]))[0])
    var = float(Jx @ basin.cov @ Jx.T) + sigma**2
    return mu, max(var, 1e-18), Jx


def _d_intra(
    law: Law, basins: Sequence[Basin], x_tar: TargetRegion, N: float, D: float, sigma: float
) -> float:
    """Expected reduction in WITHIN-basin target variance from observing at (N, D).

    ``(1/|Xtar|) sum_k w_k ||J_k Sigma_k J_x^T||^2 / (sigma^2 + J_x Sigma_k J_x^T)``
    -- the rank-one Gauss-Newton update, which needs no y and so is exact rather
    than a fantasy.
    """
    out = 0.0
    for b in basins:
        _, var, Jx = _basin_predictive(law, b, N, D, sigma)
        Jt = np.atleast_2d(law.jacobian(b.theta, x_tar.N, x_tar.D))
        v = Jt @ b.cov @ Jx.T  # (n_tar, 1)
        out += b.weight * float(np.sum(v**2)) / var
    return out / len(x_tar)


def _d_inter(
    law: Law, basins: Sequence[Basin], x_tar: TargetRegion, N: float, D: float, sigma: float
) -> float:
    """Expected reduction in BETWEEN-basin disagreement from observing at (N, D).

    Observing ``y`` reweights the basins by their predictive likelihood at ``y``;
    the expectation over ``y`` is taken under the current scalar mixture by
    quadrature. Basin means on the target region are held fixed, which is the
    same local-linear approximation used everywhere else here.
    """
    if len(basins) < 2:
        return 0.0
    w = np.array([b.weight for b in basins])
    preds = np.stack([b.pred for b in basins])  # (K, n_tar)
    mus, vars_ = [], []
    for b in basins:
        mu, var, _ = _basin_predictive(law, b, N, D, sigma)
        mus.append(mu)
        vars_.append(var)
    mus = np.asarray(mus)
    sds = np.sqrt(np.asarray(vars_))

    mu_now = w @ preds
    v_now = float(np.sum(w[:, None] * (preds - mu_now) ** 2) / len(x_tar))

    lo = float(np.min(mus - 6 * sds))
    hi = float(np.max(mus + 6 * sds))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    ys = np.linspace(lo, hi, _QUAD)
    # (K, Q) component densities, then the mixture density for the expectation.
    comp = np.exp(-0.5 * ((ys[None, :] - mus[:, None]) / sds[:, None]) ** 2) / (
        sds[:, None] * np.sqrt(2 * np.pi)
    )
    mix = w @ comp
    post = w[:, None] * comp
    tot = post.sum(axis=0)
    good = tot > 1e-300
    post = np.where(good, post / np.where(good, tot, 1.0), w[:, None])
    # V_inter after observing each y on the grid.
    mu_post = post.T @ preds  # (Q, n_tar)
    sq = ((preds[None, :, :] - mu_post[:, None, :]) ** 2).sum(axis=2)  # (Q, K)
    v_post = (sq * post.T).sum(axis=1) / len(x_tar)  # (Q,)
    norm = float(_trapz(mix, ys))
    if norm <= 0:
        return 0.0
    expected = float(_trapz(v_post * mix, ys)) / norm
    return max(v_now - expected, 0.0)


def score_candidates(
    law: Law,
    basins: Sequence[Basin],
    x_tar: TargetRegion,
    candidates: Sequence[Candidate],
    *,
    sigma: float = DEFAULT_SIGMA,
    cost_alpha: float = 0.4,
) -> List[CandidateScore]:
    """Rank candidates by ``(dV_intra + dV_inter) / cost^alpha``, best first.

    ``cost_alpha`` defaults to the paper's single global 0.4, which it reports
    stable over 0.25-0.75; alpha = 0 overspends on expensive cells at low budget
    and alpha = 1 over-prefers cheap ones.
    """
    out: List[CandidateScore] = []
    for c in candidates:
        di = _d_intra(law, basins, x_tar, c.N, c.D, sigma)
        de = _d_inter(law, basins, x_tar, c.N, c.D, sigma)
        cost = max(c.cost, 1e-12)
        out.append(
            CandidateScore(
                candidate=c,
                d_intra=di,
                d_inter=de,
                cost=c.cost,
                score=(di + de) / cost**cost_alpha,
            )
        )
    return sorted(out, key=lambda s: -s.score)


def plan_batch(
    law: Law,
    observed: Sequence,
    candidates: Sequence[Candidate],
    x_tar: TargetRegion,
    *,
    budget: float,
    sigma: float = DEFAULT_SIGMA,
    cost_alpha: float = 0.4,
    n_starts: int = 128,
    admit_mult: float = 2.0,
    max_picks: int = 24,
    seed: int = 0,
) -> List[Tuple[Candidate, CandidateScore, float]]:
    """Greedy sequential selection under a budget (paper algorithm 1).

    IMPORTANT: a plan beyond the first pick is a FANTASY. Picks 2..n assume each
    earlier pick returned exactly the basin-mixture mean, because the real
    outcome needs the cell to run. Pick 1 is the only entry that depends on no
    invented data; treat the tail as a sketch of where the budget goes, and
    re-plan after each cell lands. Candidates are NOT removed when selected, so
    a repeat seed of a config can be chosen (see the module docstring).
    """
    N = [float(c.N) for c in observed]
    D = [float(c.D) for c in observed]
    L = [float(c.L) for c in observed]
    spent = 0.0
    plan: List[Tuple[Candidate, CandidateScore, float]] = []
    for _ in range(max_picks):
        affordable = [c for c in candidates if spent + c.cost <= budget]
        if not affordable:
            break
        opt = multistart(law, np.array(N), np.array(D), np.array(L), n_starts=n_starts, seed=seed)
        basins = cluster_basins(
            law,
            opt,
            np.array(N),
            np.array(D),
            x_tar.points,
            sigma=sigma,
            admit_mult=admit_mult,
        )
        if not basins:
            break
        ranked = score_candidates(
            law, basins, x_tar, affordable, sigma=sigma, cost_alpha=cost_alpha
        )
        pick = ranked[0]
        spent += pick.candidate.cost
        plan.append((pick.candidate, pick, spent))
        # Fantasy observation: the basin-mixture mean at the picked config.
        mu = sum(
            b.weight
            * float(
                law.predict(b.theta, np.array([pick.candidate.N]), np.array([pick.candidate.D]))[0]
            )
            for b in basins
        )
        N.append(float(pick.candidate.N))
        D.append(float(pick.candidate.D))
        L.append(float(mu))
    return plan
