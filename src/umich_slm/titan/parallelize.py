"""Data-parallel wrap for the titan trainer: DDP (default) or FSDP2.

Single-node only — no TP/PP. ``torch.distributed`` is initialized in
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
    comm_hook: str = "none",
) -> nn.Module:
    """Wrap ``model`` in DDP if a process group is initialized.

    ``find_unused_parameters=True`` is required for MoD (some experts may not
    receive any tokens), MTP (heads only used in training), and DeltaNet
    hybrid layouts. ``static_graph=False`` is required because routing graphs
    in MoD/MoE/DeltaNet hybrid are not static across steps.
    """
    if not is_distributed():
        return model

    ddp = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
        gradient_as_bucket_view=gradient_as_bucket_view,
    )
    if comm_hook == "bf16":
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        ddp.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)
    elif comm_hook != "none":
        raise ValueError(f"unknown ddp comm_hook {comm_hook!r}; use 'none' or 'bf16'")
    return ddp


def apply_fsdp(
    model: nn.Module,
    layers,
    *,
    param_dtype: str = "fp32",
    reshard_after_forward: bool = True,
) -> nn.Module:
    """FSDP2 ``fully_shard`` on each decoder block, then on ``model`` (the root).

    Call after activation checkpointing and per-block compile and before the
    optimizer is built, so the optimizer holds the sharded DTensor params (the
    torchtitan order). Sharding is in place: ``model`` stays the same object, and
    ``unwrap_ddp`` returns it. The last block and the root keep their params
    gathered after forward, since backward needs them first. No ``no_sync``: grads
    are reduce-scattered every micro-batch, which keeps them sharded.
    """
    if not is_distributed():
        return model
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh("cuda" if torch.cuda.is_available() else "cpu", (get_world_size(),))
    kw = {"mesh": mesh}
    if param_dtype == "bf16":
        kw["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
    elif param_dtype != "fp32":
        raise ValueError(f"unknown fsdp_param_dtype {param_dtype!r}; use 'fp32' or 'bf16'")
    layers = list(layers)
    for i, layer in enumerate(layers):
        fully_shard(layer, reshard_after_forward=reshard_after_forward and i < len(layers) - 1, **kw)
    fully_shard(model, reshard_after_forward=False, **kw)
    return model


def is_fsdp(model: nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:  # pragma: no cover
        return False
    return isinstance(model, FSDPModule)


def clip_grad_norm_(parameters, max_norm: float) -> torch.Tensor:
    """``torch.nn.utils.clip_grad_norm_``, plus FSDP2's sharded (DTensor) grads.

    Plain tensors take the stock function unchanged. Sharded grads: each rank sums
    the squares of its own shard, one all-reduce gives the global norm, and every
    shard is scaled in place. Non-finite norms propagate exactly as in the stock
    function, so the grad-spike guard still sees them.
    """
    from torch.distributed.tensor import DTensor

    params = [p for p in parameters if p.grad is not None]
    n_dt = sum(isinstance(p.grad, DTensor) for p in params)
    if n_dt == 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    if n_dt != len(params):
        # A replicated plain grad would be counted once per rank.
        raise RuntimeError("clip_grad_norm_: mix of sharded and plain grads under FSDP")
    local = [p.grad.to_local() for p in params]
    dev = local[0].device
    sq = torch.zeros((), device=dev, dtype=torch.float32)
    for g in local:
        if g.numel():
            sq = sq + torch.linalg.vector_norm(g, dtype=torch.float32).square()
    dist.all_reduce(sq, op=dist.ReduceOp.SUM)
    total = sq.sqrt()
    coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    for g in local:
        if g.numel():
            g.mul_(coef.to(g.dtype))
    return total


def unwrap_ddp(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def barrier() -> None:
    if is_distributed():
        dist.barrier()
