"""DDP wrap for the titan trainer.

Single-node DDP only — no FSDP2/TP/PP. ``torch.distributed`` is initialized in
``torchtitan_train/launch.py`` (via torchrun env vars) and this module just
wraps the model. Single-GPU runs no-op.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_rank_zero() -> bool:
    return get_rank() == 0


def apply_ddp(
    model: nn.Module,
    *,
    local_rank: int,
    find_unused_parameters: bool = True,
    static_graph: bool = False,
    gradient_as_bucket_view: bool = True,
) -> nn.Module:
    """Wrap ``model`` in DDP if a process group is initialized.

    ``find_unused_parameters=True`` is required for MoD (some experts may not
    receive any tokens), MTP (heads only used in training), and DeltaNet
    hybrid layouts. ``static_graph=False`` is required because routing graphs
    in MoD/MoE/DeltaNet hybrid are not static across steps.
    """
    if not is_distributed():
        return model

    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
        gradient_as_bucket_view=gradient_as_bucket_view,
    )


def unwrap_ddp(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def barrier() -> None:
    if is_distributed():
        dist.barrier()
