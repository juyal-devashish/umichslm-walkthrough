"""MuonW — Muon with decoupled weight decay, plus AdamW for everything else.

This is the Moonlight recipe (*Muon is Scalable for LLM Training*, arxiv
2502.16982), the variant the optimizer literature calls "MuonW" to distinguish
it from the Hyperball-constrained MuonH in :mod:`umich_slm.optim.muon`. It is
built to be a drop-in comparison against the sweep's AdamW cells:

* **Muon** on hidden ``nn.Linear`` weights. For a weight ``W`` of shape
  ``[rows, cols]`` with gradient ``G``::

      M <- mu * M + G                       (Nesterov: D = G + mu * M)
      O  = NewtonSchulz5(D)                  (~orthogonal polar factor)
      W <- W - lr * (0.2 * sqrt(max(rows, cols)) * O + wd * W)

  The ``0.2 * sqrt(max(rows, cols))`` factor gives the update an RMS of ~0.2,
  which is what AdamW produces, so the AdamW learning rate and weight decay
  carry over unchanged. Decay is decoupled and multiplied by ``lr`` exactly as
  in ``torch.optim.AdamW``, so ``wd=0.1`` means the same thing on both sides.
* **AdamW** (standard, decoupled decay) on embeddings, the LM head, norms,
  biases, and any 2-D parameter that is not a hidden linear map: memory-bank
  token tables, 4-way routers, loop-mixer matrices. Muon is a rule for linear
  maps between hidden spaces; a table of token vectors or a 4-row classifier is
  not one. See :func:`muonw_param_groups`.

Speed. Same-shape matrices are stacked and orthogonalized in one batched
Newton-Schulz call, in bf16 on CUDA, and every elementwise step is a
``torch._foreach_*`` op. A per-parameter loop costs ~5 x 3 GEMM launches per
matrix, which on the sweep's launch-bound small widths is a real wall-clock tax.

Distributed. Under DDP every rank holds the same all-reduced gradient, so the
momentum, decay and AdamW updates are identical elementwise work on every rank.
Newton-Schulz is the expensive part and the only one where bitwise agreement is
not guaranteed (bf16 GEMMs), so with ``ns_shard=True`` (the default once a
process group with more than one rank exists) each same-shape stack is split
across ranks, every rank orthogonalizes only its slice, and one
``all_gather_into_tensor`` per shape gives every rank the same bf16 update.
Replicas stay bitwise identical by construction and the NS work drops by the
world size, at the price of gathering the update (2 bytes per Muon param per
step). ``ns_shard=False`` orthogonalizes everything on every rank, with no
communication; it exists to time the trade-off, and nothing re-syncs replicas
if the GEMMs ever disagree.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer

from .muon import zeropower_via_newtonschulz5

__all__ = ["MuonW", "muonw_param_groups", "MUON_MIN_DIM"]

# Linear weights with a dimension below this go to AdamW. It exists to catch
# the 4 x hidden routers (mat_ffn tiers, pathways); the narrowest real
# projection in the sweep is a single-KV-head k_proj at 64 x 128 (ar18m).
MUON_MIN_DIM = 16


def muonw_param_groups(
    model: nn.Module,
    base_groups: list[dict],
    *,
    lr: float,
    lr_adam: float,
    weight_decay: float,
    min_dim: int = MUON_MIN_DIM,
) -> list[dict]:
    """Split ``model`` into ``[muon_group, *adamw_groups]``.

    ``base_groups`` is the AdamW grouping the baseline cells use
    (``UMichSLMTitanWrapper.param_groups``); the Muon weights are removed from
    it and the remainder keeps its own decay settings, so the AdamW side of a
    MuonW cell is configured exactly like the matching AdamW cell. The Muon
    group comes first so ``scheduler.get_last_lr()[0]`` is the Muon LR and
    ``[1]`` the AdamW LR (``train.py`` logs that as ``lr_adam``).
    """
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    head_ids = {id(head.weight)} if head is not None else set()
    muon: list[nn.Parameter] = []
    muon_ids: set[int] = set()
    for mod in model.modules():
        if not isinstance(mod, nn.Linear):
            continue
        w = mod.weight
        if not w.requires_grad or id(w) in head_ids or id(w) in muon_ids:
            continue
        if min(w.shape) < min_dim:
            continue
        muon.append(w)
        muon_ids.add(id(w))
    groups = [
        {"params": muon, "use_muon": True, "lr": lr, "weight_decay": weight_decay}
    ]
    for g in base_groups:
        rest = [p for p in g["params"] if id(p) not in muon_ids]
        if rest:
            groups.append({**g, "params": rest, "use_muon": False, "lr": lr_adam})
    seen = sum(len(g["params"]) for g in groups)
    expected = sum(len(g["params"]) for g in base_groups)
    if seen != expected:
        raise ValueError(
            f"muonw_param_groups: {expected - seen:+d} params differ between the Muon split "
            "and base_groups; a Muon weight is missing from base_groups."
        )
    return [g for g in groups if g["params"]]


class MuonW(Optimizer):
    """Muon (decoupled weight decay) on ``use_muon`` groups, AdamW on the rest.

    Every param group must set ``use_muon``. Hyperparameters may be overridden
    per group; ``lr`` and ``weight_decay`` usually are.

    Parameters
    ----------
    lr, weight_decay
        Defaults for groups that do not set them. Decay is decoupled and scaled
        by ``lr`` on both sides, as in ``torch.optim.AdamW``.
    momentum, nesterov, ns_steps
        Muon momentum coefficient, Nesterov look-ahead, Newton-Schulz steps.
    update_scale
        Muon update multiplier on top of ``sqrt(max(rows, cols))``. 0.2 matches
        AdamW's update RMS (Moonlight).
    betas, eps
        AdamW moments and epsilon for ``use_muon=False`` groups.
    ns_dtype
        Newton-Schulz compute dtype. ``None`` = bf16 on CUDA, float32 on CPU.
    ns_shard
        Split Newton-Schulz across the ranks of ``process_group`` and all-gather
        the result (see the module docstring). Ignored with one rank or no
        initialized process group.
    process_group
        Group to shard over; ``None`` = the default group.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        update_scale: float = 0.2,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        ns_dtype: Optional[torch.dtype] = None,
        ns_shard: bool = True,
        process_group: Optional["dist.ProcessGroup"] = None,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"MuonW: lr must be > 0, got {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"MuonW: weight_decay must be >= 0, got {weight_decay}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"MuonW: momentum out of range: {momentum}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"MuonW: betas out of range: {betas}")
        if ns_steps < 1:
            raise ValueError(f"MuonW: ns_steps must be >= 1, got {ns_steps}")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            update_scale=update_scale,
            betas=tuple(betas),
            eps=eps,
        )
        super().__init__(params, defaults)
        self.ns_dtype = ns_dtype
        self.ns_shard = bool(ns_shard)
        self.process_group = process_group
        for g in self.param_groups:
            if "use_muon" not in g:
                raise ValueError(
                    "MuonW: every param group must set use_muon (see muonw_param_groups)"
                )
            if g["use_muon"]:
                for p in g["params"]:
                    if p.ndim != 2:
                        raise ValueError(
                            f"MuonW: use_muon groups take 2-D weights only, got shape "
                            f"{tuple(p.shape)}"
                        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("MuonW does not support sparse gradients")
            if group["use_muon"]:
                self._muon_step(group, params)
            else:
                self._adamw_step(group, params)
        return loss

    def _muon_step(self, group: dict, params: list[torch.Tensor]) -> None:
        lr, wd, mu = group["lr"], group["weight_decay"], group["momentum"]
        grads = [p.grad for p in params]
        bufs = []
        for p in params:
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)
            bufs.append(state["momentum_buffer"])
        torch._foreach_mul_(bufs, mu)
        torch._foreach_add_(bufs, grads)
        dirs = torch._foreach_add(grads, bufs, alpha=mu) if group["nesterov"] else bufs
        if wd != 0.0:
            torch._foreach_mul_(params, 1.0 - lr * wd)
        by_shape: dict[tuple, list[int]] = defaultdict(list)
        for i, p in enumerate(params):
            by_shape[(tuple(p.shape), p.dtype, p.device)].append(i)
        world, rank = self._world()
        for (shape, dtype, device), idx in by_shape.items():
            ns_dtype = self.ns_dtype or (
                torch.bfloat16 if device.type == "cuda" else torch.float32
            )
            if world > 1:
                ortho = self._sharded_ns(dirs, idx, shape, device, ns_dtype, group, world, rank)
            else:
                stacked = torch.stack([dirs[i] for i in idx]).to(ns_dtype)
                ortho = zeropower_via_newtonschulz5(stacked, steps=group["ns_steps"])
            ortho = ortho.to(dtype)
            scale = group["update_scale"] * math.sqrt(max(shape))
            torch._foreach_add_([params[i] for i in idx], list(ortho.unbind(0)), alpha=-lr * scale)

    def _world(self) -> tuple[int, int]:
        """(world, rank) to shard Newton-Schulz over; (1, 0) when not sharding."""
        if not self.ns_shard or not (dist.is_available() and dist.is_initialized()):
            return 1, 0
        world = dist.get_world_size(self.process_group)
        return (world, dist.get_rank(self.process_group)) if world > 1 else (1, 0)

    def _sharded_ns(self, dirs, idx, shape, device, ns_dtype, group, world, rank):
        """Orthogonalize rank ``rank``'s contiguous slice of one same-shape stack
        and all-gather the slices, so every rank returns the same ``[len(idx), *shape]``.

        Slices are ceil(n / world) matrices; the tail is zero-padded (a zero
        direction orthogonalizes to zero and is dropped after the gather).
        """
        k = -(-len(idx) // world)
        mine = idx[rank * k : (rank + 1) * k]
        local = torch.zeros((k, *shape), dtype=ns_dtype, device=device)
        if mine:
            stacked = torch.stack([dirs[i] for i in mine]).to(ns_dtype)
            local[: len(mine)] = zeropower_via_newtonschulz5(stacked, steps=group["ns_steps"])
        full = torch.empty((k * world, *shape), dtype=ns_dtype, device=device)
        if dist.get_backend(self.process_group) == "nccl":
            dist.all_gather_into_tensor(full, local, group=self.process_group)
        else:  # gloo (CPU tests): list form, gathered straight into views of `full`
            dist.all_gather(list(full.chunk(world)), local, group=self.process_group)
        return full[: len(idx)]

    def _adamw_step(self, group: dict, params: list[torch.Tensor]) -> None:
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        beta1, beta2 = group["betas"]
        grads = [p.grad for p in params]
        exp_avgs, exp_avg_sqs, steps = [], [], []
        for p in params:
            state = self.state[p]
            if "exp_avg" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            steps.append(state["step"])
        if wd != 0.0:
            torch._foreach_mul_(params, 1.0 - lr * wd)
        torch._foreach_lerp_(exp_avgs, grads, 1.0 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)
        # Same formula as torch.optim.AdamW (non-amsgrad):
        #   p -= lr / bc1 * m / (sqrt(v) / sqrt(bc2) + eps)
        denom = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(denom, [math.sqrt(1.0 - beta2**s) for s in steps])
        torch._foreach_add_(denom, eps)
        torch._foreach_addcdiv_(
            params, exp_avgs, denom, [-lr / (1.0 - beta1**s) for s in steps]
        )
