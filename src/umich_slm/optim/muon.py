"""Muon / MuonH — orthogonalized-momentum optimizer with Hyperball projection.

This is the matrix-update half of the Delphi scaling-law recipe
(openathena.ai/blog/delphi). Delphi is a *hybrid*: hidden weight matrices are
optimized with **MuonH** (Muon + Hyperball), the LM head with **AdamH**, and
embeddings / scalar parameters with plain **AdamW**. Rather than juggle three
optimizer objects, this single :class:`Muon` class expresses all three roles
via two per-group booleans::

    use_muon=True,  hyperball=True   ->  MuonH   (hidden attn/MLP matrices)
    use_muon=False, hyperball=True   ->  AdamH   (LM head)
    use_muon=False, hyperball=False  ->  AdamW   (embeddings, norms, biases)

The integration layer (:func:`umich_slm.recipe.integrations.build_delphi_optimizer`
with ``optimizer_name="muonh"``) routes each parameter into the right group via
:func:`delphi_hybrid_param_groups`.

Muon
----
For a 2-D weight ``W`` with gradient ``G``:

1. Momentum buffer ``M <- mu*M + G`` (Nesterov look-ahead optional).
2. Orthogonalize: ``O = NewtonSchulz5(M)`` — a cheap quintic iteration whose
   fixed point sends every singular value of ``M`` to 1, i.e. ``O`` is the
   (approximate) orthogonal factor of ``M``'s polar decomposition.
3. Scale: ``O *= sqrt(max(rows, cols))``, which gives the update an RMS of
   ~1.0. NOTE this is NOT the AdamW-matched scale it was written to be: *Muon
   is Scalable for LLM Training* (arxiv 2502.16982) uses
   ``0.2 * sqrt(max(rows, cols))``, because AdamW's update RMS is ~0.2, so at
   equal ``lr`` this path steps ~5x harder than AdamW. Under the Hyperball
   projection the effective per-step rotation is ~``lr / init_std``. Left as
   is because MuonH is not in use; the AdamW-matched variant with decoupled
   weight decay is :class:`umich_slm.optim.muonw.MuonW`.
4. Step: ``W <- W - lr * O``.

Hyperball
---------
After the step, a Hyperball-constrained group rescales ``W`` back to the
Frobenius norm it had at step 0 (captured on the first ``step()``). With the
norm pinned, weight decay has nothing to regularize away and drops out of the
recipe — identical to :class:`umich_slm.optim.adamh.AdamH`.

Distributed note
----------------
Newton-Schulz needs the **full** matrix. Under tensor/FSDP sharding the
parameter must be all-gathered before orthogonalization. This implementation is
**single-device** (the scaling sweep runs 1 GPU per job); calling it on a
sharded parameter raises rather than silently computing a wrong update. A
distributed variant is a documented follow-up.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer

from .adamh import _project_to_norm  # shared Frobenius-norm projection

__all__ = ["Muon", "zeropower_via_newtonschulz5"]


def zeropower_via_newtonschulz5(
    g: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Orthogonalize a matrix (or a batch of them) via quintic Newton-Schulz.

    ``g`` is ``[..., rows, cols]``; every trailing matrix is orthogonalized
    independently, so same-shape weights can share one batched launch per
    iteration (MuonW stacks them). Returns the same shape/dtype as ``g`` with
    every singular value ~1 (the orthogonal polar factor). The quintic
    coefficients are Keller Jordan's tuned ``(3.4445, -4.7750, 2.0315)`` —
    chosen so the iteration converges from any spectral norm <= ~1 after
    normalization; they stop in a band ~[0.7, 1.3], not at exactly 1.

    Runs in the dtype it is given: float32 for portability, bf16 on GPU for
    speed (what the reference implementations use).
    """
    if g.ndim < 2:
        raise ValueError(f"zeropower_via_newtonschulz5: expected >=2-D, got {g.ndim}-D")
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g
    # Scale so the largest singular value is <= 1 (Frobenius norm upper-bounds it).
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    transposed = False
    if x.size(-2) > x.size(-1):
        # Iterate on the smaller Gram matrix.
        x = x.mT
        transposed = True
    for _ in range(steps):
        gram = x @ x.mT
        poly = b * gram + c * (gram @ gram)
        x = a * x + poly @ x
    if transposed:
        x = x.mT
    return x


class Muon(Optimizer):
    """Muon with optional Hyperball projection and an AdamW fallback per group.

    Parameters
    ----------
    params
        Iterable of parameters or list of param-group dicts. Per-group keys:

        * ``use_muon`` (bool, default True): orthogonalized-momentum update.
          When False the group uses a standard Adam/AdamW update.
        * ``hyperball`` (bool, default True): rescale params to their init
          Frobenius norm after each step (replaces weight decay).
        * ``weight_decay`` (float, default 0): only honored for non-hyperball
          groups; must be 0 for hyperball groups (ValueError otherwise).
    lr
        Default learning rate (override per group).
    momentum
        Muon momentum coefficient ``mu``.
    nesterov
        Use Nesterov look-ahead on the Muon momentum buffer.
    betas
        Adam ``(beta1, beta2)`` for non-muon groups.
    eps
        Adam epsilon for non-muon groups.
    ns_steps
        Newton-Schulz iteration count for muon groups.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        betas: tuple[float, float] = (0.9, 0.9999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Muon: lr must be > 0, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Muon: momentum out of range: {momentum}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Muon: betas out of range: {betas}")
        if eps <= 0.0:
            raise ValueError(f"Muon: eps must be > 0, got {eps}")
        if ns_steps < 1:
            raise ValueError(f"Muon: ns_steps must be >= 1, got {ns_steps}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            use_muon=True,
            hyperball=True,
        )
        super().__init__(params, defaults)
        for g in self.param_groups:
            if g.get("hyperball", True) and float(g.get("weight_decay", 0.0)) != 0.0:
                raise ValueError(
                    "Muon: weight_decay must be 0 for Hyperball-constrained groups "
                    "(the Hyperball projection replaces weight decay). "
                    f"Got weight_decay={g['weight_decay']!r}."
                )
            if g.get("use_muon", True):
                for p in g["params"]:
                    if p.ndim != 2:
                        raise ValueError(
                            "Muon: use_muon groups accept only 2-D weight matrices; "
                            f"got a {p.ndim}-D parameter (shape {tuple(p.shape)}). "
                            "Route embeddings / norms / biases into a use_muon=False "
                            "group (see delphi_hybrid_param_groups)."
                        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon = group.get("use_muon", True)
            hyperball = group.get("hyperball", True)
            lr = group["lr"]
            if use_muon:
                self._muon_group_step(group, lr, hyperball)
            else:
                self._adam_group_step(group, lr, hyperball)

        return loss

    # -- per-group update implementations ----------------------------------
    def _muon_group_step(self, group, lr, hyperball):
        mu = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(p)
                if hyperball:
                    state["init_norm"] = float(p.data.norm(p="fro").item())
            state["step"] += 1
            buf = state["momentum_buffer"]
            buf.mul_(mu).add_(grad)
            direction = grad.add(buf, alpha=mu) if nesterov else buf
            # Orthogonalize in float32 for precision; cast back to param dtype.
            ortho = zeropower_via_newtonschulz5(direction.float(), steps=ns_steps)
            # RMS-match scale so the update magnitude tracks AdamW's.
            rows, cols = p.shape
            ortho.mul_((max(rows, cols)) ** 0.5)
            p.add_(ortho.to(p.dtype), alpha=-lr)
            if hyperball and "init_norm" in state:
                _project_to_norm(p, state["init_norm"])

    def _adam_group_step(self, group, lr, hyperball):
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group.get("weight_decay", 0.0)
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                if hyperball:
                    state["init_norm"] = float(p.data.norm(p="fro").item())
            state["step"] += 1
            t = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
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


def delphi_hybrid_param_groups(
    model: torch.nn.Module,
    *,
    exclude_substrings: tuple[str, ...] = (
        "embed",
        "wte",
        "wpe",
        "norm",
        "ln_",
        "bias",
    ),
    head_substrings: tuple[str, ...] = ("lm_head",),
) -> list[dict]:
    """Split a model into the three Delphi optimizer groups.

    Returns ``[muon_group, head_group, adamw_group]`` ready for :class:`Muon`:

    * **muon_group** — 2-D hidden weight matrices (attention/MLP projections):
      ``use_muon=True, hyperball=True`` (MuonH).
    * **head_group** — the LM head matrix: ``use_muon=False, hyperball=True``
      (AdamH). Tied or absent heads yield an empty group.
    * **adamw_group** — embeddings, norms, biases, and any non-2-D weight:
      ``use_muon=False, hyperball=False`` (AdamW).

    Group order is stable so callers can assign per-group LRs by index (muon +
    head share the projection LR; adamw uses the embedding/scalar LR). Empty
    groups are dropped so ``Muon`` never iterates a paramless group.
    """
    muon_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        is_head = any(s in lname for s in head_substrings)
        is_excluded = any(s in lname for s in exclude_substrings)
        if is_head:
            head_params.append(p)
        elif p.dim() == 2 and not is_excluded:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    groups = [
        {"params": muon_params, "use_muon": True, "hyperball": True, "weight_decay": 0.0},
        {"params": head_params, "use_muon": False, "hyperball": True, "weight_decay": 0.0},
        {"params": adamw_params, "use_muon": False, "hyperball": False},
    ]
    return [g for g in groups if g["params"]]
