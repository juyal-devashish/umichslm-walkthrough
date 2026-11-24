"""AdamH — Adam with Hyperball projection.

Implements the optimizer used in the Delphi scaling-law recipe
(openathena.ai/blog/delphi). The update rule is identical to Adam except
that, after each step, every parameter tensor in a *Hyperball-constrained*
group is rescaled to its initial Frobenius norm. The norm constraint is what
lets the paper drop weight decay from the hyperparameter search entirely.

Design notes
------------

* The Hyperball constraint is applied only to 2-D (or higher) projection
  weights — embeddings, scalar parameters (norms, biases), and the LM head
  are *not* norm-preserved. Caller controls this by routing parameters into
  the right group via ``param_groups`` with ``hyperball=True`` / ``False``.

* The reference norm is captured on the **first** ``step()`` call. This
  means model construction can use any init (1/√d_in, Xavier, etc.) and
  AdamH will preserve whatever was there at step 0.

* ``weight_decay`` is forced to 0 for any Hyperball-constrained group; a
  ValueError is raised at construction if the caller passes anything else.
  Non-Hyperball groups can still set weight_decay (this lets you decay
  embeddings if you really want).

* Two separate LR groups are expected by the Delphi recipe — projection
  (``hyperball=True``) and scalar/embedding (``hyperball=False``) — with
  different horizon scalings. AdamH itself only cares about the flag; LR
  scheduling lives in :mod:`umich_slm.recipe.delphi`.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import torch
from torch.optim import Optimizer


def _project_to_norm(p: torch.Tensor, target_norm: float) -> None:
    """In-place: rescale ``p`` so ``||p||_F == target_norm``."""
    current = p.data.norm(p="fro")
    if current.item() == 0.0 or target_norm == 0.0:
        return
    p.data.mul_(target_norm / current)


class AdamH(Optimizer):
    """Adam with Hyperball (Frobenius-norm-preserving) projection.

    Parameters
    ----------
    params
        Iterable of parameters or list of ``dict`` param groups. Each group
        may set ``hyperball`` (bool, default True) controlling whether the
        norm-preservation step is applied to its parameters.
    lr
        Default learning rate. Override per-group as usual.
    betas
        Adam (β₁, β₂).
    eps
        Adam numerical-stability ε.
    weight_decay
        Must be 0 for Hyperball-constrained groups. Raises ValueError
        otherwise.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.9999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"AdamH: lr must be > 0, got {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"AdamH: beta1 out of range: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"AdamH: beta2 out of range: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"AdamH: eps must be > 0, got {eps}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hyperball=True,
        )
        super().__init__(params, defaults)
        # Validate: hyperball groups must have weight_decay == 0.
        for g in self.param_groups:
            if g.get("hyperball", True) and float(g.get("weight_decay", 0.0)) != 0.0:
                raise ValueError(
                    "AdamH: weight_decay must be 0 for Hyperball-constrained groups "
                    "(the Hyperball projection replaces weight decay). "
                    f"Got weight_decay={g['weight_decay']!r}."
                )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group.get("weight_decay", 0.0)
            hyperball = group.get("hyperball", True)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamH does not support sparse gradients")

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if hyperball:
                        # Capture init norm on the first step.
                        state["init_norm"] = float(p.data.norm(p="fro").item())

                state["step"] += 1
                t = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if wd != 0.0 and not hyperball:
                    grad = grad.add(p, alpha=wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                step_size = lr / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

                if hyperball and "init_norm" in state:
                    _project_to_norm(p, state["init_norm"])

        return loss


def split_param_groups(
    model: torch.nn.Module,
    *,
    proj_min_dim: int = 2,
    exclude_substrings: tuple[str, ...] = (
        "embed",
        "wte",
        "wpe",
        "lm_head",
        "norm",
        "ln_",
        "bias",
    ),
) -> list[dict]:
    """Split ``model.parameters()`` into Hyperball / non-Hyperball groups.

    The convention matches the Delphi recipe: projection matrices (≥ 2-D
    parameters whose name doesn't match an embedding/norm/bias pattern) live
    in the Hyperball-constrained group; everything else is non-constrained.

    The returned list can be passed directly to :class:`AdamH`. The two
    groups carry distinct ``hyperball`` flags but no learning rate — caller
    is expected to set per-group LR (e.g. via
    :func:`umich_slm.recipe.delphi.lr_groups_for_model`).
    """
    proj_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_proj = (
            p.dim() >= proj_min_dim
            and not any(s in name.lower() for s in exclude_substrings)
        )
        (proj_params if is_proj else other_params).append(p)
    return [
        {"params": proj_params, "hyperball": True, "weight_decay": 0.0},
        {"params": other_params, "hyperball": False, "weight_decay": 0.0},
    ]
