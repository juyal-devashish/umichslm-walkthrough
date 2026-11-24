"""Delphi-style training recipe utilities (LR, schedule, β₂, ε, arch).

Importing this module is side-effect-free; nothing is auto-registered. The
public API is exposed in :mod:`umich_slm.recipe.delphi`.
"""

from .delphi import (
    DelphiRecipeConfig,
    beta2_for_batch,
    build_wsd_schedule,
    compute_horizon_lr,
    delphi_arch,
    epsilon_for_bt,
    lr_groups_for_model,
)

__all__ = [
    "DelphiRecipeConfig",
    "beta2_for_batch",
    "build_wsd_schedule",
    "compute_horizon_lr",
    "delphi_arch",
    "epsilon_for_bt",
    "lr_groups_for_model",
]
