"""Grad-spike guard: skip poisoned optimizer steps without wedging the run.

Extracted from the trainer so the recovery policy is unit-testable. Three arms
were lost to the previous inline version (swa_deltabank 10m/50m, think_suffix
ar10m) and one of them still wrote a summary.json whose val_lm_loss read as a
mediocre architecture rather than a model frozen at 52% of its schedule.

The failure mode is NOT a divergence. In every case the pre-clip grad norm
jumped once (0.25 -> 44, 2.5 -> 91) and then sat *stable* at the new scale
forever while lm_loss stayed flat to four decimals. The guard was not absorbing
a spike; it was refusing to follow a permanent change in gradient SCALE, and
because a skipped step never updates the EMA, the refusal was self-sustaining.

Policy:
  * one-off spike            -> skip (unchanged; this is the case it was built for)
  * sustained skip streak    -> adopt the new scale as the EMA and resume
  * streak that survives N adoptions -> raise, so the cell dies loudly

Adopting is safe because the trainer clips to max_grad_norm before applying, so
a large raw norm cannot produce a large update. Skipping was strictly the more
conservative option and it is the one that lost the runs.
"""

from __future__ import annotations

import math
from typing import Optional


class GradSpikeGuard:
    def __init__(
        self,
        *,
        enabled: bool = True,
        factor: float = 20.0,
        warmup: int = 100,
        recover_after: int = 50,
        max_rewarms: int = 5,
        ema_decay: float = 0.99,
    ) -> None:
        self.enabled = bool(enabled)
        self.factor = float(factor)
        self.warmup = int(warmup)
        self.recover_after = int(recover_after)
        self.max_rewarms = int(max_rewarms)
        self.ema_decay = float(ema_decay)

        self.ema: Optional[float] = None
        self.skips_total = 0
        self.consec_skips = 0
        self.rewarms = 0
        # Set on the step a re-seed happens so the trainer can log it.
        self.last_event: Optional[str] = None

    def check(self, grad_norm: float, steps_done: int) -> bool:
        """Return True if this step's optimizer update should be SKIPPED.

        ``steps_done`` is steps completed since the loop start (resume-relative),
        matching the original arming condition.
        """
        self.last_event = None
        if not self.enabled:
            return False
        nonfinite = not math.isfinite(grad_norm)
        armed = self.ema is not None and steps_done >= self.warmup
        spike = armed and grad_norm > self.factor * self.ema
        if not (nonfinite or spike):
            return False

        self.skips_total += 1
        self.consec_skips += 1
        # A non-finite norm is never adopted: re-seeding the EMA to inf/nan would
        # silently disable the guard for the rest of the run.
        if (
            self.recover_after > 0
            and self.consec_skips >= self.recover_after
            and math.isfinite(grad_norm)
        ):
            if self.rewarms >= self.max_rewarms:
                raise RuntimeError(
                    f"grad-spike guard: {self.consec_skips} consecutive skips after "
                    f"{self.rewarms} EMA re-seeds (grad_norm={grad_norm:.3e}, "
                    f"ema={self.ema:.3e}). Run is wedged; aborting rather than "
                    f"training on forfeited steps and writing a frozen summary."
                )
            self.rewarms += 1
            self.last_event = (
                f"RE-SEED after {self.consec_skips} consecutive skips: "
                f"ema {self.ema:.3e} -> {grad_norm:.3e} "
                f"({self.rewarms}/{self.max_rewarms})"
            )
            self.ema = grad_norm
            self.consec_skips = 0
        return True

    def applied(self, grad_norm: float) -> None:
        """Record an APPLIED step. Only healthy, applied norms move the EMA."""
        self.consec_skips = 0
        if math.isfinite(grad_norm):
            self.ema = (
                grad_norm
                if self.ema is None
                else (self.ema_decay * self.ema + (1.0 - self.ema_decay) * grad_norm)
            )
