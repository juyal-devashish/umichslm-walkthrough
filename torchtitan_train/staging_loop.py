"""In-process multi-stage driver for layer-stacked pretraining.

When ``[staging].mode == "in_process"``, the launcher dispatches here instead
of calling :func:`torchtitan_train.train.run` directly. This driver:

  1. Runs stage 1 by calling :func:`run` once with the original config (the
     model is built from ``[model].size``).
  2. Between stages, calls :func:`umich_slm.staging.transition.run_transition`
     in-process — same code as the standalone CLI used by LLaMA-Factory — to
     grow the previous stage's checkpoint into the new stage's directory.
  3. Re-invokes :func:`run` with a config that points ``[model].init_from`` at
     the grown directory, ``[checkpoint].resume_from`` to load the migrated
     optimizer state, and a per-stage ``[training].max_steps`` /
     ``[optimizer].lr`` if the user provided per-stage values.

The disk round-trip between stages is intentional:
  * It keeps the stage-N+1 transition reproducible (the grown checkpoint can
    be inspected, the optimizer migration is auditable via
    ``stage_metadata.json``).
  * It reuses the existing single-stage ``run`` function unchanged, avoiding a
    risky in-place refactor of a 400-line training loop.
  * On 8×H100, the wall-clock cost of saving + reloading a 1B model is small
    relative to a stage's training time (minutes vs hours).

Compared to the LLaMA-Factory multi-process flow, the only difference is that
here the entire chain runs in a single launch (single sbatch job) rather than
as a chain of dependent sbatch jobs.
"""

from __future__ import annotations

import copy
import logging
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

# Path setup, identical to train.py — so this file works as a script too.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_PATH = _REPO_ROOT / "src"
if _SRC_PATH.exists() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

logger = logging.getLogger(__name__)


def _stage_subdir(output_dir: Path, stage_index: int) -> Path:
    return output_dir / f"stage_{stage_index}"


def _stage_init_subdir(output_dir: Path, stage_index: int) -> Path:
    return output_dir / f"stage_{stage_index}_init"


def _stage_max_steps(staging_cfg, training_cfg, stage_idx: int, total_stages: int) -> int:
    """Resolve the max_steps for the given stage.

    If ``[staging].max_steps_per_stage`` is provided, use the corresponding
    entry; otherwise fall back to ``[training].max_steps`` for every stage.
    """
    per_stage = list(staging_cfg.max_steps_per_stage or [])
    if per_stage and len(per_stage) >= total_stages:
        return int(per_stage[stage_idx])
    return int(training_cfg.max_steps)


def _stage_lr(staging_cfg, optimizer_cfg, stage_idx: int, total_stages: int) -> float:
    per_stage = list(staging_cfg.lr_per_stage or [])
    if per_stage and len(per_stage) >= total_stages:
        return float(per_stage[stage_idx])
    return float(optimizer_cfg.lr)


def run_staged(config, *, rank: int, local_rank: int, world_size: int) -> None:
    """Multi-stage driver. Runs ``len(config.staging.stages)`` stages back-to-back.

    Args:
        config: Parsed ``TitanConfig`` whose ``[staging].mode == "in_process"``.
        rank, local_rank, world_size: distributed rank info from the launcher.
    """
    from umich_slm.staging.block_spec import StageSpec
    from umich_slm.staging.transition import run_transition
    from torchtitan_train.train import run

    if config.staging.mode != "in_process":
        raise ValueError(
            f"run_staged() requires staging.mode='in_process'; got {config.staging.mode!r}."
        )

    stage_paths = list(config.staging.stages or [])
    n_stages = len(stage_paths)
    if n_stages < 1:
        raise ValueError("staging.stages must list at least one StageSpec JSON path.")

    output_dir = Path(config.job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load all stage specs and validate up-front to fail fast on bad JSON.
    stage_specs = []
    for sp in stage_paths:
        spec = StageSpec.load(sp)
        stage_specs.append(spec)

    prev_ckpt_dir: Optional[Path] = None

    for stage_idx, spec_path in enumerate(stage_paths):
        spec = stage_specs[stage_idx]
        stage_index_label = spec.stage_index

        # Per-stage clone of the top-level config so we can rewrite paths and LR.
        stage_config = copy.deepcopy(config)

        # Each stage writes into its own subdir for clean checkpoint provenance.
        stage_output = _stage_subdir(output_dir, stage_index_label)
        stage_config.job.output_dir = str(stage_output)
        stage_config.job.run_name = f"{config.job.run_name}_stage_{stage_index_label}"

        # Stage-specific budgets (max_steps, lr).
        stage_config.training.max_steps = _stage_max_steps(
            config.staging, config.training, stage_idx, n_stages
        )
        stage_config.optimizer.lr = _stage_lr(
            config.staging, config.optimizer, stage_idx, n_stages
        )

        # Per-stage TST opt-in: copy from StageSpec.tst into the stage's
        # TitanConfig.training.tst. If the stage spec doesn't set TST, leave
        # the top-level config's TST setting alone (so a user can also enable
        # TST globally via the TOML if every stage should use the same s/r).
        if spec.tst is not None:
            stage_tst = stage_config.training.tst
            stage_tst.enabled = bool(spec.tst.enabled)
            stage_tst.bag_size = int(spec.tst.bag_size)
            stage_tst.recovery_ratio = float(spec.tst.recovery_ratio)
            stage_tst.loss_weighting = str(spec.tst.loss_weighting)
            stage_tst.extend_seq_len_for_equal_flops = bool(
                spec.tst.extend_seq_len_for_equal_flops
            )

        if stage_idx == 0:
            # First stage uses the model defined by [model].size (or init_from
            # if the user supplied a bootstrap). The stage spec is set on the
            # config so that the model is constructed with the correct
            # hybrid_attention_ratio, attention_mode, and stage_block_spec.
            stage_config.model.overrides = {
                **(stage_config.model.overrides or {}),
                "attention_mode": spec.attention_mode,
                "hybrid_attention_ratio": list(spec.hybrid_attention_ratio),
                "stage_block_spec": [bs.to_dict() for bs in spec.block_specs],
                "num_hidden_layers": spec.num_hidden_layers,
            }
            logger.info(
                "[staging] Running stage %d in %s (model from preset, max_steps=%d, lr=%g)",
                stage_index_label, stage_output,
                stage_config.training.max_steps, stage_config.optimizer.lr,
            )
        else:
            assert prev_ckpt_dir is not None
            # Run the in-process transition: writes a grown HF checkpoint plus
            # migrated optimizer.pt into a sibling init dir.
            init_dir = _stage_init_subdir(output_dir, stage_index_label)
            if init_dir.exists() and not (init_dir / "_DONE").exists():
                # Half-finished previous attempt: clear it.
                shutil.rmtree(init_dir)
            run_transition(
                from_dir=prev_ckpt_dir,
                to_dir=init_dir,
                spec_path=Path(spec_path),
                init_strategy=config.staging.init_strategy,
                run_probes=config.staging.run_probes_between_stages,
            )

            # Stage N >= 2 reads the grown checkpoint as init_from and resumes
            # the migrated optimizer.
            stage_config.model.init_from = str(init_dir)
            # Remove model.overrides for stage 2+ — the grown checkpoint already
            # has the right config baked in.
            stage_config.model.overrides = {}
            stage_config.checkpoint.resume_from = str(init_dir)

            logger.info(
                "[staging] Running stage %d in %s (init_from=%s, max_steps=%d, lr=%g)",
                stage_index_label, stage_output, init_dir,
                stage_config.training.max_steps, stage_config.optimizer.lr,
            )

        # Run this stage. The training loop saves a final checkpoint to
        # stage_config.job.output_dir/checkpoint-{step}; we point the next
        # stage at the most recently-written checkpoint dir.
        run(stage_config, rank=rank, local_rank=local_rank, world_size=world_size)

        # Resolve the final checkpoint dir for the next stage. We pick the
        # checkpoint dir with the highest numeric suffix.
        prev_ckpt_dir = _resolve_final_checkpoint(stage_output)
        if prev_ckpt_dir is None:
            raise RuntimeError(
                f"No checkpoint found under {stage_output} after stage "
                f"{stage_index_label}; expected checkpoint-* subdirectory."
            )
        logger.info("[staging] Stage %d final checkpoint: %s", stage_index_label, prev_ckpt_dir)

    logger.info("[staging] All %d stages complete.", n_stages)


def _resolve_final_checkpoint(stage_output: Path) -> Optional[Path]:
    """Return the highest-step checkpoint-N subdirectory under ``stage_output``.

    Falls back to ``stage_output`` itself if checkpoints were saved in-place.
    """
    if not stage_output.exists():
        return None
    candidates = []
    for entry in stage_output.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        # Match "checkpoint-12345" or "checkpoint_12345" patterns.
        for sep in ("checkpoint-", "checkpoint_"):
            if name.startswith(sep):
                try:
                    step = int(name[len(sep):])
                    candidates.append((step, entry))
                    break
                except ValueError:
                    continue
    if candidates:
        candidates.sort()
        return candidates[-1][1]
    # Fallback: assume the stage_output itself holds the HF files.
    if (stage_output / "config.json").exists():
        return stage_output
    return None
