"""Token-Superposition Training (TST) phase controller.

Recovery cutover is a single boolean flip on ``model.config.tst_active``. This
file exposes the logic in three forms so all three pretraining stacks share it:

- ``TSTController``: plain object, used inline by the torchtitan loop.
- ``TSTTrainerCallback``: HF ``TrainerCallback`` wrapper, mirrors
  ``DRoPECallback`` in style.
- ``maybe_advance_tst(controller, step, max_steps, model)``: free function for
  loops that prefer not to instantiate the callback.

The controller flips ``model.config.tst_active`` from True→False at step
``int(max_steps * recovery_ratio)``. It does not touch the data loader; the
training loop is expected to slice batches back to the standard sequence
length once ``tst_active`` becomes False (see torchtitan_train/train.py).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _flip_active(model, active: bool) -> None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("TSTController: model has no .config attribute")
    cfg.tst_active = bool(active)


class TSTController:
    """Tracks the TST → recovery cutover step and toggles ``model.config.tst_active``.

    Args:
        enabled: Master switch. If False, this controller is a no-op.
        bag_size: Bag size ``s`` for phase (i).
        recovery_ratio: Fraction ``r`` of total steps spent in phase (i).
            The cutover happens at step ``int(max_steps * r)``.
        verbose: Log on phase transitions.
    """

    def __init__(
        self,
        enabled: bool,
        bag_size: int,
        recovery_ratio: float,
        verbose: bool = True,
    ):
        if enabled and not 0.0 < recovery_ratio < 1.0:
            raise ValueError(
                f"recovery_ratio must be in (0, 1) when enabled, got {recovery_ratio}"
            )
        if enabled and bag_size < 2:
            raise ValueError(f"bag_size must be >= 2 when enabled, got {bag_size}")
        self.enabled = bool(enabled)
        self.bag_size = int(bag_size)
        self.recovery_ratio = float(recovery_ratio)
        self.verbose = bool(verbose)
        self._initialized = False
        self._cutover_step: Optional[int] = None
        self._currently_active: bool = False

    @property
    def cutover_step(self) -> Optional[int]:
        return self._cutover_step

    @property
    def active(self) -> bool:
        return self._currently_active

    def begin(self, max_steps: int, model) -> None:
        """Call once at the start of a stage. Sets cutover step and flips flag on."""
        if not self.enabled:
            _flip_active(model, False)
            return
        if max_steps is None or max_steps <= 0:
            raise ValueError(f"TSTController.begin: invalid max_steps={max_steps}")
        self._cutover_step = int(max_steps * self.recovery_ratio)
        # Ensure config carries the bag size used by the model's forward.
        model.config.tst_enabled = True
        model.config.tst_bag_size = self.bag_size
        _flip_active(model, True)
        self._currently_active = True
        self._initialized = True
        if self.verbose:
            logger.info(
                "TST phase (i) activated: bag_size=%d, recovery at step %d / %d",
                self.bag_size,
                self._cutover_step,
                max_steps,
            )

    def step(self, global_step: int, model) -> bool:
        """Check whether the cutover should fire on this step. Returns the new
        ``active`` state (True = still phase i, False = phase ii / recovery).
        """
        if not self.enabled or not self._initialized:
            return False
        if self._currently_active and global_step >= (self._cutover_step or 0):
            _flip_active(model, False)
            self._currently_active = False
            if self.verbose:
                logger.info(
                    "TST recovery: switching to standard NTP at step %d", global_step
                )
        return self._currently_active

    def end(self, model) -> None:
        """Ensure the flag is off at end of stage (safety: subsequent stages /
        eval / generation must not run with TST active)."""
        _flip_active(model, False)
        self._currently_active = False


def maybe_advance_tst(controller: Optional[TSTController], step: int, model) -> bool:
    """Convenience for inline loops. Returns whether TST phase (i) is still active."""
    if controller is None:
        return False
    return controller.step(step, model)


# ---------------------------------------------------------------------------
# HF Trainer callback wrapper
# ---------------------------------------------------------------------------

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState
    from transformers.training_args import TrainingArguments

    class TSTTrainerCallback(TrainerCallback):
        """HF TrainerCallback that drives a :class:`TSTController`.

        Mirrors the pattern in ``callbacks/drope.py``: compute the cutover step
        from ``state.max_steps`` in ``on_train_begin`` and flip the flag at
        ``on_step_end``.
        """

        def __init__(
            self,
            enabled: bool,
            bag_size: int,
            recovery_ratio: float,
            verbose: bool = True,
        ):
            self.controller = TSTController(
                enabled=enabled,
                bag_size=bag_size,
                recovery_ratio=recovery_ratio,
                verbose=verbose,
            )

        def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            model=None,
            **kwargs,
        ):
            if model is None or state.max_steps is None or state.max_steps <= 0:
                logger.warning(
                    "TSTTrainerCallback: missing model or max_steps; TST will not activate"
                )
                return
            self.controller.begin(state.max_steps, model)

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            model=None,
            **kwargs,
        ):
            if model is None:
                return
            self.controller.step(state.global_step, model)

        def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            model=None,
            **kwargs,
        ):
            if model is None:
                return
            self.controller.end(model)

except ImportError:  # pragma: no cover
    TSTTrainerCallback = None  # type: ignore[assignment]
