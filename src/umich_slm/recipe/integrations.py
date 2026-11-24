"""Integration adapters that wire the Delphi recipe into our three stacks.

The three training stacks each construct an optimizer + LR scheduler in a
different idiom (HF Trainer, LLaMA-Factory, torchtitan). To keep their
recipe code in lockstep, all three call into the helpers below rather than
re-implementing the formulas. The shared code lives in
:mod:`umich_slm.recipe.delphi`; this file is the thin glue.

Each helper takes a :class:`DelphiRecipeConfig` plus stack-specific objects
and returns the configured (optimizer, scheduler) pair. The helpers never
mutate the model.

Failure mode policy: each helper raises a ``RuntimeError`` with a clear
message if the recipe asks for something the stack can't provide
(e.g. AdamH requested but :mod:`umich_slm.optim` not importable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .delphi import (
    DelphiRecipeConfig,
    beta2_for_batch,
    build_schedule,
    epsilon_for_bt,
    hybrid_lr_groups_for_model,
    lr_groups_for_model,
)

if TYPE_CHECKING:
    import torch


def build_delphi_optimizer(
    model: "torch.nn.Module",
    *,
    tokens: float,
    batch: int,
    config: Optional[DelphiRecipeConfig] = None,
    optimizer_name: str = "adamh",
) -> "torch.optim.Optimizer":
    """Construct the Delphi optimizer (AdamH by default) for a model.

    Caller specifies the run's (tokens, batch) so β₂/ε/LR can be computed
    from the formulas. The returned optimizer carries two param groups
    (projection / scalar+embedding) with distinct LRs.
    """
    cfg = config or DelphiRecipeConfig()
    beta2 = beta2_for_batch(batch, cfg)
    eps = epsilon_for_bt(batch=batch, tokens=tokens, config=cfg)

    name = (optimizer_name or "adamh").lower()
    if name == "muonh":
        # Full Delphi hybrid: MuonH (hidden) + AdamH (head) + AdamW (embed/scalar).
        try:
            from ..optim.muon import Muon
        except ImportError as e:
            raise RuntimeError(
                "build_delphi_optimizer: MuonH requested but umich_slm.optim.muon "
                "is unavailable. " + str(e)
            )
        groups = hybrid_lr_groups_for_model(model, tokens=tokens, batch=batch, config=cfg)
        return Muon(groups, betas=(cfg.beta1, beta2), eps=eps)

    groups = lr_groups_for_model(model, tokens=tokens, batch=batch, config=cfg)
    if name == "adamh":
        try:
            from ..optim.adamh import AdamH
        except ImportError as e:
            raise RuntimeError(
                "build_delphi_optimizer: AdamH requested but umich_slm.optim.adamh "
                "is unavailable. " + str(e)
            )
        return AdamH(groups, betas=(cfg.beta1, beta2), eps=eps)
    if name == "adamw":
        import torch

        return torch.optim.AdamW(groups, betas=(cfg.beta1, beta2), eps=eps)
    raise ValueError(f"build_delphi_optimizer: unknown optimizer {optimizer_name!r}")


def build_delphi_scheduler(
    optimizer: "torch.optim.Optimizer",
    *,
    total_steps: int,
    config: Optional[DelphiRecipeConfig] = None,
) -> Any:
    """Return a torch ``LambdaLR`` matching the recipe's schedule."""
    import torch

    cfg = config or DelphiRecipeConfig()
    lam = build_schedule(
        cfg.schedule,
        total_steps=total_steps,
        warmup_frac=cfg.warmup_frac,
        decay_frac=cfg.decay_frac,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)


# ---------------------------------------------------------------------------
# HF Trainer adapter
# ---------------------------------------------------------------------------
def hf_trainer_optim_factories(
    model: "torch.nn.Module",
    *,
    tokens: float,
    batch: int,
    total_steps: int,
    config: Optional[DelphiRecipeConfig] = None,
    optimizer_name: str = "adamh",
) -> tuple:
    """Return ``(optimizer, lr_scheduler)`` for ``Trainer.optimizers``.

    HF's ``Trainer`` accepts a pre-built ``(optimizer, scheduler)`` tuple via
    its ``optimizers`` constructor kwarg. Use this from a custom Trainer
    subclass or via the ``create_optimizer_and_scheduler`` hook.
    """
    opt = build_delphi_optimizer(
        model, tokens=tokens, batch=batch, config=config, optimizer_name=optimizer_name
    )
    sched = build_delphi_scheduler(opt, total_steps=total_steps, config=config)
    return opt, sched


# ---------------------------------------------------------------------------
# LLaMA-Factory adapter (registry stub)
# ---------------------------------------------------------------------------
def llamafactory_register_delphi() -> None:
    """Best-effort registration of AdamH + WSD with LLaMA-Factory.

    LF builds optimizers via its ``OptimizerFactory``; we monkey-patch a
    handler in. This function is safe to call multiple times. If LF isn't
    importable (it's an optional dep) the function is a no-op so importing
    this module never breaks tests.
    """
    try:
        import llamafactory  # noqa: F401
    except Exception:
        return
    # Real registration would hook into llamafactory.train.trainer; we
    # expose the helper as a public callable so a future LF version can
    # call into it. Today this is a documented placeholder.


# ---------------------------------------------------------------------------
# torchtitan adapter
# ---------------------------------------------------------------------------
def torchtitan_build_optimizer_and_scheduler(
    model: "torch.nn.Module",
    *,
    tokens: float,
    batch: int,
    total_steps: int,
    config: Optional[DelphiRecipeConfig] = None,
    optimizer_name: str = "adamh",
) -> tuple:
    """Mirror of :func:`hf_trainer_optim_factories` for the torchtitan loop.

    torchtitan owns its own training loop, so the loop's
    ``build_optimizer`` callback should delegate here when the TOML config
    sets ``[recipe].kind = "delphi"``.
    """
    return hf_trainer_optim_factories(
        model, tokens=tokens, batch=batch, total_steps=total_steps,
        config=config, optimizer_name=optimizer_name,
    )
