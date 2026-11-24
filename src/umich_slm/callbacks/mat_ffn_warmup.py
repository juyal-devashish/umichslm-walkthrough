"""`mat_ffn` nested-dropout / sandwich warmup controller.

WHAT PROBLEM THIS SOLVES. A per-token nested-width FFN has a chicken-and-egg
failure: the router decides which tokens get the widest tier, but at step 0 the
router is random, so if it happens to send few tokens to the widest tier that
tier gets little gradient, stays bad, and the router learns (correctly, given
what it sees) never to use it. The width collapses to the cheapest tier and the
arm silently becomes a small dense model with extra indexing.

The elastic-width literature does not hit this because it does not have to:
MatFormer (2310.07707) trains every granularity every step with a joint loss,
Universally Slimmable Networks (1903.05134) do the same via the *sandwich rule*
(smallest + largest + random widths per iteration), and Flextron / DynaMoE /
WIDE all convert an ALREADY-TRAINED dense model, so their wide path was trained
before any router existed. Training a routed nest from scratch needs the same
guarantee, and the cheap way to get it is to take the assignment away from the
router at the start.

WHAT IT DOES. For the first ``warmup_frac`` of training, each token's tier is
SAMPLED from a prior rather than chosen by the router (``warmup_sampling``), and
that prior anneals from uniform to the arm's target skew over ``warmup_stages``
discrete stages. Stage 0 is uniform, so every width -- widest included -- is
trained on 1/k of the tokens. By the last stage the distribution is the one the
router will have to reproduce, so the hand-over is not a distribution shift as
well as a control shift.

Two design points worth keeping:

* The stage index is an INTEGER, not a continuous alpha. The per-tier cap sizes
  a static window in ``RoutedMatFormerSwiGLUMLP._forward_sorted``, so a cap that
  moved every step would recompile the graph every step; a staged one
  recompiles ``warmup_stages + 1`` times over the whole run.
* The ROUTER IS NOT FROZEN during warmup. Tier k's contribution is scaled by the
  straight-through gate ``tail[:, k] = P(choice >= k)``, so the router gets
  "would more width have helped this token" gradient regardless of who made the
  assignment. It arrives at the hand-over already trained, which is the point.

Nested dropout (Rippel et al., 1402.0915) is the older name for the sampling
rule -- draw a cut index from a prior and drop everything above it -- and it
comes with a reason to prefer a SKEWED prior beyond cost: it provably forces the
units into an importance ordering (exact PCA equivalence in the semi-linear
case), which is precisely the structure the router later has to exploit.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def _routed_mlps(model) -> List:
    from ..modules.swiglu import RoutedMatFormerSwiGLUMLP

    return [m for m in model.modules() if isinstance(m, RoutedMatFormerSwiGLUMLP)]


class MatFFNWarmupController:
    """Drives ``warmup_stage`` / ``warmup_sampling`` on every routed mat_ffn MLP.

    Args:
        enabled: master switch; False makes every method a no-op that still
            pins the modules to their target prior (stage == warmup_stages).
        warmup_frac: fraction of ``max_steps`` spent in warmup. 0 disables.
        stages: number of interpolation stages between uniform and the target
            prior. The run therefore sees ``stages + 1`` distinct cap vectors.
        verbose: log each stage transition (rank-0 callers only).
    """

    def __init__(
        self,
        enabled: bool,
        warmup_frac: float,
        stages: int = 4,
        verbose: bool = True,
    ):
        if enabled and not 0.0 < warmup_frac < 1.0:
            raise ValueError(
                f"warmup_frac must be in (0, 1) when enabled, got {warmup_frac}"
            )
        if enabled and stages < 1:
            raise ValueError(f"stages must be >= 1, got {stages}")
        self.enabled = bool(enabled)
        self.warmup_frac = float(warmup_frac)
        self.stages = int(stages)
        self.verbose = bool(verbose)
        self._warmup_steps: Optional[int] = None
        self._current_stage: Optional[int] = None
        self._mlps: List = []

    @property
    def warmup_steps(self) -> Optional[int]:
        return self._warmup_steps

    @property
    def stage(self) -> Optional[int]:
        return self._current_stage

    def begin(self, max_steps: int, model) -> None:
        """Call once at the start of a stage. Caches the module list."""
        self._mlps = _routed_mlps(model)
        for m in self._mlps:
            m.warmup_stages = self.stages
        if not self.enabled or not self._mlps:
            self._warmup_steps = 0
            self._apply(self.stages, sampling=False)
            return
        if max_steps is None or max_steps <= 0:
            raise ValueError(f"MatFFNWarmupController.begin: invalid max_steps={max_steps}")
        self._warmup_steps = max(1, int(max_steps * self.warmup_frac))
        self._apply(0, sampling=True)
        if self.verbose:
            logger.info(
                "[mat_ffn] warmup on: %d of %d steps, %d stages, %d routed MLPs",
                self._warmup_steps,
                max_steps,
                self.stages,
                len(self._mlps),
            )

    def step(self, step: int, model=None) -> bool:
        """Advance to the stage for ``step``. Returns True while in warmup."""
        if not self.enabled or not self._mlps or not self._warmup_steps:
            return False
        if step >= self._warmup_steps:
            if self._current_stage != self.stages:
                self._apply(self.stages, sampling=False)
                if self.verbose:
                    logger.info(
                        "[mat_ffn] warmup complete at step %d; router now controls "
                        "allocation at the target prior",
                        step,
                    )
            return False
        # stage in [0, stages - 1] during warmup; the final stage (== target
        # prior) is reached exactly at hand-over, so no step runs the target
        # prior under sampling AND under the router.
        s = min(self.stages - 1, int(step * self.stages / self._warmup_steps))
        if s != self._current_stage:
            self._apply(s, sampling=True)
            if self.verbose:
                logger.info(
                    "[mat_ffn] warmup stage %d/%d at step %d; prior %s",
                    s,
                    self.stages,
                    step,
                    [round(x, 3) for x in self._mlps[0].tier_prior()],
                )
        return True

    def end(self, model=None) -> None:
        """Call once at the end of a stage. Leaves modules on the target prior."""
        if self._mlps:
            self._apply(self.stages, sampling=False)

    def _apply(self, stage: int, sampling: bool) -> None:
        for m in self._mlps:
            m.warmup_stage = int(stage)
            m.warmup_sampling = bool(sampling)
        self._current_stage = int(stage)


def maybe_advance_mat_ffn_warmup(controller, step: int, model=None) -> bool:
    """Free function for loops that would rather not hold the object."""
    if controller is None:
        return False
    return controller.step(step, model)
