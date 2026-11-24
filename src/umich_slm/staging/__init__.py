"""Staging package for layer-stacked pretraining.

The same module is consumed by:
  * :mod:`torchtitan_train.train` — in-process stage transitions during a
    single launch (``[staging].mode = "in_process"``).
  * The standalone CLI ``python -m umich_slm.staging.transition`` — used
    between LLaMA-Factory sbatch jobs to grow a checkpoint from stage N-1
    into stage N.
"""

from .block_spec import (
    BlockSpec,
    StageSpec,
    StageTSTSpec,
    stage_spec_from_dict,
    stage_spec_to_dict,
)
from .build_stage import build_stage_n_model
from .flops_accounting import (
    chinchilla_optimal_params,
    chinchilla_optimal_tokens,
    effective_tokens,
    final_params,
    stage_points,
    total_flops,
)
from .io_boundary import recommend_boundary
from .optimizer_migration import migrate_optimizer_state

__all__ = [
    "BlockSpec",
    "StageSpec",
    "StageTSTSpec",
    "build_stage_n_model",
    "chinchilla_optimal_params",
    "chinchilla_optimal_tokens",
    "effective_tokens",
    "final_params",
    "migrate_optimizer_state",
    "recommend_boundary",
    "stage_points",
    "stage_spec_from_dict",
    "stage_spec_to_dict",
    "total_flops",
]
