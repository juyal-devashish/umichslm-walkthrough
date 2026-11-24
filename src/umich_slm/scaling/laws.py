"""Parametric scaling-law families, with analytic Jacobians.

The Jacobian is the point of this module. Every piece of machinery in
``basins.py`` and ``design.py`` -- the Gauss-Newton local covariance, the
prediction-space clustering, both halves of the acquisition function -- is built
on ``dpred/dtheta``, so a finite-difference Jacobian would make the whole stack
both slow and quietly inaccurate in the flat directions that matter most. Those
flat directions are exactly where our fits live: in-sample RMSE on the A40 grid
is 0.0018-0.0063 nats, *below* the 0.0127 seed floor, while fitted ``E`` spans
0.0000-1.1600. ``tests/test_scaling_design.py`` checks every analytic Jacobian
against central differences.

Two families are provided:

``ead``
    ``L = E + A*N^-alpha + B*D^-beta`` -- what ``scripts/fit_a40_grid.py`` has
    always fitted, kept bit-identical (same parameterisation, same bounds, raw
    N and D) so that previously reported numbers reproduce.

``farseer``
    ``L = exp(s*N^q + S) + exp(B*N^b + Q) * D^-exp(A*N^a + E)``, from Farseer
    (Li et al. 2025b, arXiv 2506.10972), as used by the ``farseer`` task of
    Li et al. (2026), arXiv 2604.22753. Its data exponent is a FUNCTION of N,
    which is the only form here that can answer the standing question of whether
    beta varies with N -- our TPR ladder exists at ar10m only, so every beta we
    quote is pinned at one N and then carried across a 55x range up to ~1B.

Both scale their inputs internally (N/1e6, D/1e9 for ``farseer``; raw for
``ead``). For ``farseer`` that is a pure reparameterisation -- the scale factor
is absorbed by the additive constants S, Q and E -- but it keeps ``N^q`` away
from overflow when q drifts toward its bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np

__all__ = ["Law", "LAWS", "get_law"]


@dataclass(frozen=True)
class Law:
    """A scaling-law family.

    ``predict(theta, N, D)`` and ``jacobian(theta, N, D)`` both broadcast over
    array-valued N and D; ``jacobian`` returns shape ``(..., n_params)``.
    """

    name: str
    params: Tuple[str, ...]
    lower: Tuple[float, ...]
    upper: Tuple[float, ...]
    predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    jacobian: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    sample_x0: Callable[[np.random.Generator], np.ndarray]
    # Human-readable derived quantities, e.g. the data exponent at a given N.
    report: Callable[[np.ndarray, float], Dict[str, float]]
    # Parameters whose positivity we respect by rescaling Jacobian columns
    # (paper appendix C.1: equivalent to measuring sensitivity in log space).
    positive: Tuple[str, ...] = ()
    # Parameters PINNED to a value, as ``((index, value), ...)``. A restricted
    # law is expressed this way rather than by collapsing its bounds, because
    # scipy's least_squares requires every lower bound to be STRICTLY below its
    # upper bound and raises otherwise -- which, swallowed by a multi-start loop,
    # silently yields zero optima instead of a restricted fit.
    fixed: Tuple[Tuple[int, float], ...] = ()

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def n_free(self) -> int:
        return len(self.free_idx)

    @property
    def free_idx(self) -> Tuple[int, ...]:
        pinned = {i for i, _ in self.fixed}
        return tuple(i for i in range(self.n_params) if i not in pinned)

    def with_fixed(self, free_theta: np.ndarray) -> np.ndarray:
        """Expand a free-parameter vector to the full one, reinserting pinned values."""
        full = np.empty(self.n_params)
        full[list(self.free_idx)] = np.asarray(free_theta, dtype=float)
        for i, v in self.fixed:
            full[i] = v
        return full

    def bounds(self) -> Tuple[Sequence[float], Sequence[float]]:
        return list(self.lower), list(self.upper)

    def free_bounds(self) -> Tuple[Sequence[float], Sequence[float]]:
        idx = list(self.free_idx)
        return [self.lower[i] for i in idx], [self.upper[i] for i in idx]


# --------------------------------------------------------------------- ead
# Parameterised as log A / log B so the scale params stay positive without
# bounds fighting the solver; alpha/beta bounded to (0, 1] and E to a
# non-negative entropy floor. This mirrors fit_a40_grid.py exactly.
_EAD_PARAMS = ("E", "logA", "alpha", "logB", "beta")
_EAD_LO = (0.0, -50.0, 1e-3, -50.0, 1e-3)
_EAD_HI = (np.inf, 50.0, 1.0, 50.0, 1.0)


def _ead_predict(theta, N, D):
    E, logA, alpha, logB, beta = theta
    return E + np.exp(logA) * N ** (-alpha) + np.exp(logB) * D ** (-beta)


def _ead_jacobian(theta, N, D):
    E, logA, alpha, logB, beta = theta
    N = np.asarray(N, dtype=float)
    D = np.asarray(D, dtype=float)
    A, B = np.exp(logA), np.exp(logB)
    tN, tD = N ** (-alpha), D ** (-beta)
    return np.stack(
        [
            np.ones_like(N),  # dE
            A * tN,  # d logA
            -A * tN * np.log(N),  # d alpha
            B * tD,  # d logB
            -B * tD * np.log(D),  # d beta
        ],
        axis=-1,
    )


def _ead_x0(rng):
    return np.array(
        [
            rng.uniform(0.0, 2.5),  # E
            rng.uniform(1.0, 20.0),  # log A
            rng.uniform(0.05, 0.6),  # alpha
            rng.uniform(1.0, 20.0),  # log B
            rng.uniform(0.05, 0.6),  # beta
        ]
    )


def _ead_report(theta, N):
    E, logA, alpha, logB, beta = theta
    return {
        "E": float(E),
        "alpha": float(alpha),
        "beta": float(beta),
        "A": float(np.exp(logA)),
        "B": float(np.exp(logB)),
    }


EAD = Law(
    name="ead",
    params=_EAD_PARAMS,
    lower=_EAD_LO,
    upper=_EAD_HI,
    predict=_ead_predict,
    jacobian=_ead_jacobian,
    sample_x0=_ead_x0,
    report=_ead_report,
    positive=("E", "alpha", "beta"),
)


# ----------------------------------------------------------------- farseer
# L = exp(s*Nh^q + S) + exp(B*Nh^b + Q) * Dh^-exp(A*Nh^a + E),
# with Nh = N/1e6 and Dh = D/1e9. The irreducible term, the data coefficient and
# the DATA EXPONENT are each their own function of N; beta is constant in N iff
# A == 0 or a == 0, which is what makes the A==0 restriction a clean nested test.
_FARSEER_PARAMS = ("s", "q", "S", "B", "b", "Q", "A", "a", "E")
_FARSEER_LO = (-50.0, -2.0, -20.0, -50.0, -2.0, -20.0, -50.0, -2.0, -20.0)
_FARSEER_HI = (50.0, 2.0, 5.0, 50.0, 2.0, 20.0, 50.0, 2.0, 5.0)
_N0, _D0 = 1e6, 1e9
# Exponent clip. Loss is O(1), so anything past exp(60) is already a dead region
# of the search; clipping here (rather than at 700) keeps the SQUARED residual
# finite, which is what scipy's trust-region step needs to stay numerically sane.
_CLIP = 60.0


def _farseer_parts(theta, N, D):
    """Pieces of the farseer prediction, plus a mask per clipped exponent.

    The clip keeps the squared residual finite for scipy's trust-region step, but
    a clipped exponent is a FLAT region: perturbing the parameters behind it does
    not move the prediction, so the derivative there is exactly zero. Returning
    the masks keeps ``jacobian`` consistent with ``predict`` -- without them the
    analytic derivative disagrees with a finite difference wherever the search
    strays into the clip, which is precisely where it is hardest to notice.
    """
    s, q, S, B, b, Q, A, a, E = theta
    Nh = np.asarray(N, dtype=float) / _N0
    Dh = np.asarray(D, dtype=float) / _D0
    lnN, lnD = np.log(Nh), np.log(Dh)
    Nq, Nb, Na = Nh**q, Nh**b, Nh**a
    raw1, raw2, raw3 = s * Nq + S, B * Nb + Q, A * Na + E
    live1 = (raw1 > -_CLIP) & (raw1 < _CLIP)
    live2 = (raw2 > -_CLIP) & (raw2 < _CLIP)
    live3 = (raw3 > -_CLIP) & (raw3 < _CLIP)
    T1 = np.exp(np.clip(raw1, -_CLIP, _CLIP))
    coef = np.exp(np.clip(raw2, -_CLIP, _CLIP))
    expo = np.exp(np.clip(raw3, -_CLIP, _CLIP))
    G = coef * Dh ** (-expo)
    return (Nh, Dh, lnN, lnD, Nq, Nb, Na, T1, coef, expo, G, live1, live2, live3)


def _farseer_predict(theta, N, D):
    parts = _farseer_parts(theta, N, D)
    return parts[7] + parts[10]


def _farseer_jacobian(theta, N, D):
    s, q, S, B, b, Q, A, a, E = theta
    (_, _, lnN, lnD, Nq, Nb, Na, T1, _, expo, G, live1, live2, live3) = _farseer_parts(
        theta, N, D
    )
    # d/dtheta of exp(s*Nh^q + S), zero where that exponent is clipped.
    d_s = T1 * Nq * live1
    d_q = T1 * s * Nq * lnN * live1
    d_S = T1 * live1
    # d/dtheta of exp(B*Nh^b + Q) * Dh^-expo
    d_B = G * Nb * live2
    d_b = G * B * Nb * lnN * live2
    d_Q = G * live2
    # expo enters as Dh^-expo, so d/d expo = -lnD * G; chain through A, a, E.
    dG_dexpo = -lnD * G
    d_A = dG_dexpo * expo * Na * live3
    d_a = dG_dexpo * expo * A * Na * lnN * live3
    d_E = dG_dexpo * expo * live3
    return np.stack([d_s, d_q, d_S, d_B, d_b, d_Q, d_A, d_a, d_E], axis=-1)


def _farseer_x0(rng):
    return np.array(
        [
            rng.uniform(-2.0, 2.0),  # s
            rng.uniform(-1.0, 1.0),  # q
            rng.uniform(-3.0, 1.0),  # S
            rng.uniform(-2.0, 2.0),  # B
            rng.uniform(-1.0, 1.0),  # b
            rng.uniform(-2.0, 2.0),  # Q
            rng.uniform(-1.0, 1.0),  # A
            rng.uniform(-1.0, 1.0),  # a
            rng.uniform(-3.0, 0.0),  # E
        ]
    )


def _farseer_report(theta, N):
    """Derived quantities at one N. ``beta`` here is the LOCAL data exponent."""
    s, q, S, B, b, Q, A, a, E = theta
    Nh = float(N) / _N0
    return {
        "beta_at_N": float(np.exp(np.clip(A * Nh**a + E, -_CLIP, _CLIP))),
        "E_at_N": float(np.exp(np.clip(s * Nh**q + S, -_CLIP, _CLIP))),
        "coef_at_N": float(np.exp(np.clip(B * Nh**b + Q, -_CLIP, _CLIP))),
    }


FARSEER = Law(
    name="farseer",
    params=_FARSEER_PARAMS,
    lower=_FARSEER_LO,
    upper=_FARSEER_HI,
    predict=_farseer_predict,
    jacobian=_farseer_jacobian,
    sample_x0=_farseer_x0,
    report=_farseer_report,
)

# `farseer_fixedbeta` is `farseer` with A pinned to 0, i.e. a data exponent that
# does NOT vary with N. Nested inside `farseer`, so the pair is the beta(N) test:
# fit both on the same cells and compare TARGET-REGION error, never in-sample.
def _farseer_fixedbeta_x0(rng):
    x = _farseer_x0(rng)
    x[6] = 0.0
    return x


FARSEER_FIXEDBETA = Law(
    name="farseer_fixedbeta",
    params=_FARSEER_PARAMS,
    lower=_FARSEER_LO,
    upper=_FARSEER_HI,
    predict=_farseer_predict,
    jacobian=_farseer_jacobian,
    sample_x0=_farseer_fixedbeta_x0,
    report=_farseer_report,
    fixed=((6, 0.0),),
)


LAWS: Dict[str, Law] = {law.name: law for law in (EAD, FARSEER, FARSEER_FIXEDBETA)}


def get_law(name: str) -> Law:
    try:
        return LAWS[name]
    except KeyError:
        raise SystemExit(f"unknown law {name!r}; choose from {sorted(LAWS)}")
