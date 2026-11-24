"""Scaling-law fitting judged where the decision lives, not in sample.

The entry points are:

``laws``
    Parametric families with analytic Jacobians -- ``ead`` (our
    ``E + A/N^alpha + B/D^beta``) and ``farseer``, whose data exponent is a
    function of N and so can test whether beta varies with N.

``basins``
    The basin-mixture posterior: every converged local optimum, clustered in
    prediction space over the target region, BIC-weighted, with the target-region
    MSPE split into within-basin (``V_intra``) and between-basin (``V_inter``)
    parts.

``design``
    Target-region metrics, the cheap/expensive-half design diagnostic, and the
    cost-aware acquisition used to choose which cells to run next.

Method from Li et al. (2026), "Spend Less, Fit Better: Budget-Efficient Scaling
Law Fitting via Active Experiment Selection", arXiv 2604.22753 (COLM 2026), with
two deliberate departures documented in ``design``: our measured seed sigma
drives the curvature, and repeat seeds are legal candidates.
"""

from .basins import (
    DEFAULT_SIGMA,
    Basin,
    Optimum,
    cluster_basins,
    local_covariance,
    mspe,
    multistart,
    predict_mixture,
)
from .design import (
    DEEP80_N_ACTIVE,
    Candidate,
    CandidateScore,
    TargetRegion,
    design_diagnostic,
    plan_batch,
    score_candidates,
    target_metrics,
)
from .laws import LAWS, Law, get_law

__all__ = [
    "Basin",
    "Candidate",
    "CandidateScore",
    "DEEP80_N_ACTIVE",
    "DEFAULT_SIGMA",
    "LAWS",
    "Law",
    "Optimum",
    "TargetRegion",
    "cluster_basins",
    "design_diagnostic",
    "get_law",
    "local_covariance",
    "mspe",
    "multistart",
    "plan_batch",
    "predict_mixture",
    "score_candidates",
    "target_metrics",
]
