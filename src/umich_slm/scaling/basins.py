"""Basin-mixture posterior over scaling-law parameters.

Why this exists. ``scripts/fit_a40_grid.py`` used to run 200 multi-starts and
keep only ``argmin(cost)``, which throws away the only information that says how
much the fit's TARGET-REGION prediction is actually pinned down. Our in-sample
RMSE sits below the 0.0127-nat seed floor while fitted ``E`` spans 0.0000-1.1600
across arms, so "the fit is good" and "the extrapolation is determined" are
different claims and we were only ever checking the first.

This implements the basin machinery of Li et al. (2026), arXiv 2604.22753,
appendix C: every converged local optimum gets a Gauss-Newton local covariance,
optima are clustered in PREDICTION space over the target region (so two
parameter vectors that extrapolate identically are one basin however far apart
they sit in parameter space), and basins are weighted BIC-style. The target-region
mean squared prediction error then splits into

    MSPE_tar = V_intra + V_inter

where ``V_intra`` is within-basin predictive variance on the target region and
``V_inter`` is disagreement BETWEEN basins there. ``V_inter`` is the quantity we
had no way to compute, and a large one means the cells in hand do not determine
the prediction no matter how small the residuals are.

One deliberate departure from the paper. The paper admits every local optimum it
finds. We have a measured seed noise floor (sigma = 0.0127 nats), so any fit
whose in-sample RMSE is at or below sigma is indistinguishable from the best fit
BY OUR OWN STANDARD and has an equal claim to being the right basin. Basin
admission therefore uses ``max(sigma, best_rmse * mult)`` rather than a pure
multiple of the best, which is strictly more inclusive whenever the grid is
fitted better than the noise floor -- i.e. always, for us.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import least_squares
from scipy.spatial.distance import squareform

from .laws import Law

__all__ = [
    "Optimum",
    "Basin",
    "DEFAULT_SIGMA",
    "multistart",
    "local_covariance",
    "cluster_basins",
    "mspe",
    "predict_mixture",
]

# Seed noise floor in nats, measured on this project's sweep. Two cells differ
# meaningfully only past 2*sqrt(2)*sigma ~ 0.036.
DEFAULT_SIGMA = 0.0127


@dataclass
class Optimum:
    """One converged local optimum of the least-squares objective."""

    theta: np.ndarray
    mse: float

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.mse))


@dataclass
class Basin:
    """A cluster of local optima that agree on the target region."""

    theta: np.ndarray
    cov: np.ndarray
    weight: float
    mse: float
    n_members: int
    pred: np.ndarray = field(default_factory=lambda: np.empty(0))
    pred_var: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.mse))


# --------------------------------------------------------------------- fitting
def multistart(
    law: Law,
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    *,
    n_starts: int = 256,
    seed: int = 0,
    max_nfev: int = 20000,
) -> List[Optimum]:
    """Multi-start least squares, returning EVERY converged optimum.

    Deduplicated on the objective value and the parameter vector, so the common
    case of many starts falling into the same optimum does not inflate a basin's
    member count into a spurious mixture weight.
    """
    N = np.asarray(N, dtype=float)
    D = np.asarray(D, dtype=float)
    L = np.asarray(L, dtype=float)
    lo, hi = law.free_bounds()
    free = list(law.free_idx)
    rng = np.random.default_rng(seed)

    # Optimise the free subvector only. A pinned parameter is reinserted by
    # `with_fixed`, which keeps every bound strictly non-degenerate -- scipy
    # rejects lb == ub outright, and a multi-start loop that swallows the
    # exception would report "no fit" for a perfectly well-posed restricted law.
    def resid(free_theta):
        return law.predict(law.with_fixed(free_theta), N, D) - L

    found: List[Optimum] = []
    for _ in range(n_starts):
        x0 = np.clip(law.sample_x0(rng)[free], lo, hi)
        try:
            # A multi-start search deliberately visits bad regions, so overflow
            # inside the trust-region step is expected traffic and is already
            # handled by rejecting non-finite results below. Printing it once per
            # start buries the actual output.
            with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                r = least_squares(resid, x0, method="trf", bounds=(lo, hi), max_nfev=max_nfev)
        except Exception:
            continue
        if not np.all(np.isfinite(r.x)):
            continue
        mse = float(np.mean(resid(r.x) ** 2))
        if not np.isfinite(mse):
            continue
        theta = law.with_fixed(r.x)
        for o in found:
            if abs(o.mse - mse) <= 1e-12 * max(1.0, abs(mse)) and np.allclose(
                o.theta, theta, rtol=1e-6, atol=1e-9
            ):
                break
        else:
            found.append(Optimum(theta=theta, mse=mse))
    return sorted(found, key=lambda o: o.mse)


def local_covariance(
    law: Law,
    theta: np.ndarray,
    N: np.ndarray,
    D: np.ndarray,
    *,
    sigma: float = DEFAULT_SIGMA,
    prior_precision: float = 1e-6,
    eig_floor: float = 1e-12,
) -> np.ndarray:
    """Gauss-Newton local covariance, ``(J^T J / sigma^2 + Lambda)^-1``.

    Columns for positive-constrained parameters are rescaled by the current
    parameter value before forming the curvature (paper appendix C.1: equivalent
    to measuring sensitivity in a log parameterisation), then mapped back, which
    keeps the matrix conditioned when parameters live on very different scales --
    ours span ``E ~ 1`` against ``A = exp(logA) ~ 1e8``.
    """
    J = np.atleast_2d(law.jacobian(theta, np.asarray(N), np.asarray(D)))
    free = list(law.free_idx)
    # A pinned parameter carries no uncertainty, so it is excluded from the
    # curvature rather than regularised: leaving it in gives that direction the
    # prior's variance (1/prior_precision = 1e6 by default), which would
    # dominate V_intra with uncertainty about a parameter we are holding fixed.
    J = J[:, free]
    scale = np.ones(len(free))
    for k, i in enumerate(free):
        if law.params[i] in law.positive:
            scale[k] = max(abs(float(theta[i])), 1e-8)
    Js = J * scale
    H = Js.T @ Js / sigma**2 + prior_precision * np.eye(len(free))
    w, V = np.linalg.eigh(H)
    w = np.maximum(w, eig_floor)
    cov_scaled = (V / w) @ V.T
    S = np.diag(scale)
    cov_free = S @ cov_scaled @ S
    cov = np.zeros((law.n_params, law.n_params))
    cov[np.ix_(free, free)] = cov_free
    return cov


# ------------------------------------------------------------------ clustering
def _silhouette(dist: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette over a precomputed distance matrix. NaN if degenerate."""
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= len(labels):
        return float("nan")
    scores = []
    for i in range(len(labels)):
        same = (labels == labels[i]) & (np.arange(len(labels)) != i)
        if not same.any():
            scores.append(0.0)
            continue
        a = dist[i, same].mean()
        b = min(dist[i, labels == k].mean() for k in uniq if k != labels[i])
        denom = max(a, b)
        scores.append(0.0 if denom == 0 else (b - a) / denom)
    return float(np.mean(scores))


def _pairwise_skl(mus: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """Target-region-averaged symmetric KL between predictive Gaussians."""
    m = len(mus)
    out = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            skl = 0.25 * (
                vs[i] / vs[j]
                + vs[j] / vs[i]
                - 2.0
                + (mus[i] - mus[j]) ** 2 * (1.0 / vs[i] + 1.0 / vs[j])
            )
            out[i, j] = out[j, i] = float(np.mean(skl))
    return out


def cluster_basins(
    law: Law,
    optima: Sequence[Optimum],
    N: np.ndarray,
    D: np.ndarray,
    x_tar: np.ndarray,
    *,
    sigma: float = DEFAULT_SIGMA,
    admit_mult: float = 2.0,
    bic_temperature: float = 1.0,
    max_optima: int = 400,
) -> List[Basin]:
    """Consolidate local optima into target-region basins with BIC weights.

    ``x_tar`` is ``(n_target, 2)`` of ``(N, D)`` pairs. Admission threshold is
    ``max(sigma, best_rmse * admit_mult)`` -- see the module docstring.
    """
    if not optima:
        return []
    n_obs = len(np.asarray(N))
    best_rmse = min(o.rmse for o in optima)
    admit = max(sigma, best_rmse * admit_mult)
    keep = [o for o in optima if o.rmse <= admit][:max_optima]
    if not keep:
        keep = [optima[0]]

    tN, tD = np.asarray(x_tar, dtype=float)[:, 0], np.asarray(x_tar, dtype=float)[:, 1]
    mus, vs, covs = [], [], []
    for o in keep:
        cov = local_covariance(law, o.theta, N, D, sigma=sigma)
        Jt = np.atleast_2d(law.jacobian(o.theta, tN, tD))
        mus.append(law.predict(o.theta, tN, tD))
        vs.append(np.einsum("ij,jk,ik->i", Jt, cov, Jt) + sigma**2)
        covs.append(cov)
    mus = np.asarray(mus, dtype=float)
    vs = np.clip(np.asarray(vs, dtype=float), 1e-18, None)

    if len(keep) == 1:
        labels = np.array([1])
    else:
        dist = _pairwise_skl(mus, vs)
        if np.allclose(dist, 0.0):
            labels = np.ones(len(keep), dtype=int)
        else:
            Z = linkage(squareform(dist, checks=False), method="average")
            # Data-adaptive cut: maximise the silhouette over the merge heights,
            # as the paper does, rather than fixing a KL threshold by hand.
            best_labels, best_score = None, -np.inf
            heights = np.unique(Z[:, 2])
            cuts = np.unique(
                np.concatenate([heights, (heights[:-1] + heights[1:]) / 2.0])
                if len(heights) > 1
                else heights
            )
            for t in cuts:
                lab = fcluster(Z, t=float(t), criterion="distance")
                s = _silhouette(dist, lab)
                if np.isfinite(s) and s > best_score:
                    best_labels, best_score = lab, s
            labels = best_labels if best_labels is not None else np.ones(len(keep), dtype=int)

    basins: List[Basin] = []
    for k in np.unique(labels):
        idx = np.flatnonzero(labels == k)
        rep = int(idx[np.argmin([keep[i].mse for i in idx])])
        basins.append(
            Basin(
                theta=keep[rep].theta,
                cov=covs[rep],
                weight=0.0,
                mse=keep[rep].mse,
                n_members=len(idx),
                pred=mus[rep],
                pred_var=vs[rep],
            )
        )

    # BIC-style basin weights (paper appendix C.3, option 1). MSE is floored at
    # the noise variance: below it the log is driven by fitting noise, which
    # would hand a near-infinite weight to whichever basin happened to overfit.
    floor = sigma**2
    bics = np.array(
        [n_obs * np.log(max(b.mse, floor)) + law.n_params * np.log(max(n_obs, 2)) for b in basins]
    )
    w = np.exp(-(bics - bics.min()) / (2.0 * bic_temperature))
    w = w / w.sum()
    for b, wi in zip(basins, w):
        b.weight = float(wi)
    return sorted(basins, key=lambda b: -b.weight)


# ------------------------------------------------------------------- reporting
def predict_mixture(basins: Sequence[Basin]) -> np.ndarray:
    """Weighted mean target-region prediction across basins."""
    return sum(b.weight * b.pred for b in basins)


def mspe(
    law: Law,
    basins: Sequence[Basin],
    x_tar: np.ndarray,
    *,
    sigma: float = DEFAULT_SIGMA,
) -> Tuple[float, float, np.ndarray]:
    """Return ``(V_intra, V_inter, mixture_mean)`` over the target region."""
    if not basins:
        return float("nan"), float("nan"), np.empty(0)
    tN, tD = np.asarray(x_tar, dtype=float)[:, 0], np.asarray(x_tar, dtype=float)[:, 1]
    n_tar = len(tN)
    mu = predict_mixture(basins)
    v_intra = 0.0
    for b in basins:
        Jt = np.atleast_2d(law.jacobian(b.theta, tN, tD))
        v_intra += b.weight * float(np.trace(Jt @ b.cov @ Jt.T)) / n_tar
    v_inter = float(sum(b.weight * np.sum((b.pred - mu) ** 2) for b in basins) / n_tar)
    return v_intra, v_inter, mu
