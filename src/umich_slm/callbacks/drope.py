"""DRoPE (Dropped Rotary Position Embeddings) training callback.

This module provides a HuggingFace TrainerCallback that implements DRoPE,
a technique for training LLMs without position embeddings after an initial
training phase. This enables better length generalization.

Reference: https://arxiv.org/abs/2512.12167

Usage with HuggingFace Trainer:
    from umich_slm.callbacks import DRoPECallback
    from transformers import Trainer

    # Create callback - will drop position embeddings at 87.5% of training
    drope_callback = DRoPECallback(drop_fraction=0.875)

    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[drope_callback],
        ...
    )
    trainer.train()

Usage with LLaMA Factory:
    Since LLaMA Factory doesn't support custom callbacks directly, you can
    create a custom training script that adds the callback:

    from llamafactory.train.pt.workflow import run_pt
    from umich_slm.callbacks import DRoPECallback

    # Monkey-patch to add callback (see LLaMA Factory docs)
    # Or use the DRoPECallback.manually_toggle() method for simpler setups
"""

import logging
from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class DRoPECallback(TrainerCallback):
    """Callback to toggle DRoPE (drop position embeddings) during training.

    DRoPE training consists of two phases:
    1. Normal training with RoPE (majority of training)
    2. Recalibration without position embeddings (final portion)

    This callback automatically toggles between phases based on training progress.

    Args:
        drop_fraction: Fraction of total training (0.0-1.0) after which position
            embeddings should be dropped. Default 0.875 means RoPE is used for
            87.5% of training, then dropped for the final 12.5%.
        use_steps: If True, use global_step for progress tracking. If False,
            use epoch progress. Default True (recommended for pretraining).
        verbose: If True, log when DRoPE state changes. Default True.

    Attributes:
        drope_active: Whether position embeddings are currently dropped.
        drop_step: The step at which DRoPE will be activated (set during training).

    Example:
        >>> callback = DRoPECallback(drop_fraction=0.875)
        >>> trainer = Trainer(model=model, callbacks=[callback], ...)
        >>> trainer.train()  # DRoPE activates at 87.5% of max_steps
    """

    def __init__(
        self,
        drop_fraction: float = 0.875,
        use_steps: bool = True,
        verbose: bool = True,
    ):
        if not 0.0 < drop_fraction < 1.0:
            raise ValueError(f"drop_fraction must be in (0, 1), got {drop_fraction}")

        self.drop_fraction = drop_fraction
        self.use_steps = use_steps
        self.verbose = verbose

        # State tracking
        self.drope_active = False
        self.drop_step: Optional[int] = None
        self._initialized = False

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Initialize DRoPE tracking at the start of training."""
        if model is None:
            logger.warning("DRoPECallback: No model provided, cannot track DRoPE state")
            return

        # Calculate the step at which to drop position embeddings
        max_steps = state.max_steps
        if max_steps is None or max_steps <= 0:
            logger.warning(
                "DRoPECallback: max_steps not set, DRoPE will not activate automatically. "
                "Use manually_toggle() or set max_steps in TrainingArguments."
            )
            return

        self.drop_step = int(max_steps * self.drop_fraction)

        if self.verbose:
            logger.info(
                f"DRoPECallback initialized: will drop position embeddings at step "
                f"{self.drop_step} ({self.drop_fraction * 100:.1f}% of {max_steps} steps)"
            )

        # Ensure DRoPE starts disabled
        self._set_drope_state(model, enabled=False)
        self._initialized = True

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Check if we should activate DRoPE at this step."""
        if model is None or self.drop_step is None:
            return

        # Check if we've passed the drop point
        if not self.drope_active and state.global_step >= self.drop_step:
            self._set_drope_state(model, enabled=True)

            if self.verbose:
                logger.info(
                    f"DRoPECallback: ACTIVATED at step {state.global_step}. "
                    f"Position embeddings are now DROPPED for recalibration phase."
                )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Log DRoPE status at end of training."""
        if self.verbose:
            status = "active (no position embeddings)" if self.drope_active else "inactive"
            logger.info(f"DRoPECallback: Training ended with DRoPE {status}")

    def _set_drope_state(self, model, enabled: bool):
        """Set the DRoPE state on the model.

        Args:
            model: The model (UMichSLMModel or UMichSLMForCausalLM).
            enabled: Whether to enable DRoPE (drop position embeddings).
        """
        if hasattr(model, "set_drope_enabled"):
            model.set_drope_enabled(enabled)
            self.drope_active = enabled
        elif hasattr(model, "model") and hasattr(model.model, "set_drope_enabled"):
            # Handle wrapped models (e.g., from FSDP/DDP)
            model.model.set_drope_enabled(enabled)
            self.drope_active = enabled
        else:
            logger.warning(
                "DRoPECallback: Model does not have set_drope_enabled method. "
                "Make sure you're using a UMichSLM model."
            )

    @staticmethod
    def manually_toggle(model, enabled: bool, verbose: bool = True):
        """Manually toggle DRoPE on a model.

        This is useful for custom training loops or when not using the callback.

        Args:
            model: The model (UMichSLMModel or UMichSLMForCausalLM).
            enabled: Whether to enable DRoPE (drop position embeddings).
            verbose: Whether to log the change.

        Example:
            >>> # In your training loop:
            >>> if step >= drop_step:
            ...     DRoPECallback.manually_toggle(model, enabled=True)
        """
        if hasattr(model, "set_drope_enabled"):
            model.set_drope_enabled(enabled)
        elif hasattr(model, "model") and hasattr(model.model, "set_drope_enabled"):
            model.model.set_drope_enabled(enabled)
        else:
            raise AttributeError(
                "Model does not have set_drope_enabled method. "
                "Make sure you're using a UMichSLM model."
            )

        if verbose:
            status = "ENABLED (no position embeddings)" if enabled else "DISABLED (using RoPE)"
            logger.info(f"DRoPE manually toggled: {status}")
