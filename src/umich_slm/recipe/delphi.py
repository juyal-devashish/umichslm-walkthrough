"""Delphi scaling-law recipe: formulas and helpers.

All formulas referenced here come from openathena.ai/blog/delphi:

* **Projection LR**:   η_proj = η₀ · √(B/B₀) · (T₀/T)^horizon_exp
* **Scalar / embedding LR**:   η_adam = η₀,adam · √(B·T₀ / (B₀·T))
* **β₂ rule**:   β₂ = clip( 0.9999^(B/B₀), 0.9, 0.9999 )
* **ε rule**:    ε = ε₀ · √(B₀·T / (B·T₀))
* **Width-scaling arch**:   depth = H/(64 + 4·log₂H − 9), n_heads = H/128, MLP×4
* **Init**:   σ_proj = 1/√d_in, σ_emb = 1/H
* **Schedule**:   warmup-stable-decay (WSD) — 10% linear warmup, 70% flat,
  20% linear decay to 0.

Defaults match the published values (η₀ = 6.30e-3, B₀ = 64, T₀ = 2.5e9,
η₀,adam = 6.56e-4, ε₀ = 1.85e-8). Every constant is overridable via
:class:`DelphiRecipeConfig`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class DelphiRecipeConfig:
    """All knobs of the Delphi recipe in one place."""

    # LR formulas
    eta0_proj: float = 6.30e-3
    eta0_adam: float = 6.56e-4
    batch_ref: int = 64
    token_ref: float = 2.5e9
    horizon_exp: float = 0.30

    # Adam hyperparameters
    beta1: float = 0.9
    beta2_max: float = 0.9999
    beta2_min: float = 0.9
    epsilon0: float = 1.85e-8

    # Gradient clipping
    grad_clip: float = 0.1

    # Schedule
    warmup_frac: float = 0.10
    decay_frac: float = 0.20
    schedule: str = "wsd"  # one of {"wsd", "cosine", "linear"}

    # Architecture
    mlp_ratio: int = 4
    seq_len_default: int = 4096

    def horizon_factor(self, tokens: float) -> float:
        return (self.token_ref / max(1.0, float(tokens))) ** self.horizon_exp

    def batch_factor(self, batch: int) -> float:
        return math.sqrt(max(1, int(batch)) / float(self.batch_ref))


# ---------------------------------------------------------------------------
# Learning rate
# ---------------------------------------------------------------------------
def compute_horizon_lr(
    *,
    tokens: float,
    batch: int,
    group: str = "proj",
    config: Optional[DelphiRecipeConfig] = None,
) -> float:
    """Return the Delphi learning rate for a (tokens, batch, group) point.

    Parameters
    ----------
    tokens
        Total tokens the run will see (use effective tokens for TST runs).
    batch
        Global batch size in *sequences* (not tokens).
    group
        "proj" (Hyperball-constrained projection weights) or
        "adam" (scalar / embedding params).
    config
        Recipe constants; defaults to :class:`DelphiRecipeConfig`.
    """
    cfg = config or DelphiRecipeConfig()
    if group == "proj":
        return cfg.eta0_proj * cfg.batch_factor(batch) * cfg.horizon_factor(tokens)
    if group == "adam":
        # √(B·T₀ / (B₀·T))
        return cfg.eta0_adam * math.sqrt(
            (float(batch) * cfg.token_ref) / (float(cfg.batch_ref) * max(1.0, float(tokens)))
        )
    raise ValueError(f"compute_horizon_lr: unknown group {group!r}; expected 'proj' or 'adam'")


def lr_groups_for_model(
    model,
    *,
    tokens: float,
    batch: int,
    config: Optional[DelphiRecipeConfig] = None,
) -> list[dict]:
    """Return AdamH-ready param groups with per-group LR and Hyperball flag.

    Convenience wrapper around
    :func:`umich_slm.optim.adamh.split_param_groups` that fills in LRs.
    """
    from ..optim.adamh import split_param_groups

    cfg = config or DelphiRecipeConfig()
    proj_lr = compute_horizon_lr(tokens=tokens, batch=batch, group="proj", config=cfg)
    adam_lr = compute_horizon_lr(tokens=tokens, batch=batch, group="adam", config=cfg)
    groups = split_param_groups(model)
    # split_param_groups returns [proj_group, other_group].
    groups[0]["lr"] = proj_lr
    groups[1]["lr"] = adam_lr
    return groups


def hybrid_lr_groups_for_model(
    model,
    *,
    tokens: float,
    batch: int,
    config: Optional[DelphiRecipeConfig] = None,
) -> list[dict]:
    """Return :class:`~umich_slm.optim.muon.Muon`-ready groups for the full
    Delphi hybrid (MuonH hidden + AdamH head + AdamW embeddings/scalars).

    The MuonH hidden matrices and the AdamH head both take the **projection**
    LR (η_proj); embeddings and scalar params take the **adam** LR (η_adam) —
    matching :func:`lr_groups_for_model` but with the three-way split from
    :func:`umich_slm.optim.muon.delphi_hybrid_param_groups`.
    """
    from ..optim.muon import delphi_hybrid_param_groups

    cfg = config or DelphiRecipeConfig()
    proj_lr = compute_horizon_lr(tokens=tokens, batch=batch, group="proj", config=cfg)
    adam_lr = compute_horizon_lr(tokens=tokens, batch=batch, group="adam", config=cfg)
    groups = delphi_hybrid_param_groups(model)
    # Assign LR by role: muon (use_muon) and the hyperball head take proj LR;
    # the plain-AdamW group takes the embedding/scalar LR.
    for g in groups:
        is_proj_lr = g.get("use_muon", False) or g.get("hyperball", False)
        g["lr"] = proj_lr if is_proj_lr else adam_lr
    return groups


# ---------------------------------------------------------------------------
# Adam β₂ and ε
# ---------------------------------------------------------------------------
def beta2_for_batch(batch: int, config: Optional[DelphiRecipeConfig] = None) -> float:
    """β₂ = clip(0.9999^(B/B₀), β₂_min, β₂_max)."""
    cfg = config or DelphiRecipeConfig()
    raw = cfg.beta2_max ** (float(batch) / float(cfg.batch_ref))
    return float(min(max(raw, cfg.beta2_min), cfg.beta2_max))


def epsilon_for_bt(
    *, batch: int, tokens: float, config: Optional[DelphiRecipeConfig] = None
) -> float:
    """ε = ε₀ · √(B₀·T / (B·T₀))."""
    cfg = config or DelphiRecipeConfig()
    return cfg.epsilon0 * math.sqrt(
        (float(cfg.batch_ref) * max(1.0, float(tokens)))
        / (float(batch) * cfg.token_ref)
    )


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
def build_wsd_schedule(
    *,
    total_steps: int,
    warmup_frac: float = 0.10,
    decay_frac: float = 0.20,
) -> Callable[[int], float]:
    """Return a fraction-of-peak-LR function for warmup-stable-decay.

    The returned callable maps ``step -> multiplier in [0, 1]``. Use as
    ``LambdaLR(opt, lr_lambda=build_wsd_schedule(...))``.

    * Steps in ``[0, warmup_frac·T)`` ramp linearly from 0 to 1.
    * Steps in the middle stay at 1.
    * Steps in the last ``decay_frac·T`` decay linearly from 1 to 0.
    """
    if total_steps <= 0:
        raise ValueError(f"build_wsd_schedule: total_steps must be > 0, got {total_steps}")
    if not 0.0 <= warmup_frac <= 1.0:
        raise ValueError(f"warmup_frac out of [0,1]: {warmup_frac}")
    if not 0.0 <= decay_frac <= 1.0:
        raise ValueError(f"decay_frac out of [0,1]: {decay_frac}")
    if warmup_frac + decay_frac > 1.0:
        raise ValueError(
            f"warmup_frac+decay_frac > 1: {warmup_frac}+{decay_frac}"
        )

    warmup_end = int(round(warmup_frac * total_steps))
    decay_start = int(round((1.0 - decay_frac) * total_steps))

    def lr_lambda(step: int) -> float:
        if step <= 0:
            return 0.0
        if step < warmup_end:
            return float(step) / max(1, warmup_end)
        if step < decay_start:
            return 1.0
        if step >= total_steps:
            return 0.0
        remaining = float(total_steps - step)
        denom = max(1, total_steps - decay_start)
        return remaining / denom

    return lr_lambda


def build_cosine_schedule(
    *, total_steps: int, warmup_frac: float = 0.10
) -> Callable[[int], float]:
    """Cosine decay to 0 with linear warmup."""
    warmup_end = max(1, int(round(warmup_frac * total_steps)))

    def lr_lambda(step: int) -> float:
        if step < warmup_end:
            return float(step) / warmup_end
        if step >= total_steps:
            return 0.0
        x = (step - warmup_end) / max(1, total_steps - warmup_end)
        return 0.5 * (1.0 + math.cos(math.pi * x))

    return lr_lambda


def build_linear_schedule(
    *, total_steps: int, warmup_frac: float = 0.10
) -> Callable[[int], float]:
    """Linear decay to 0 with linear warmup."""
    warmup_end = max(1, int(round(warmup_frac * total_steps)))

    def lr_lambda(step: int) -> float:
        if step < warmup_end:
            return float(step) / warmup_end
        if step >= total_steps:
            return 0.0
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_end))

    return lr_lambda


def build_schedule(
    name: str,
    *,
    total_steps: int,
    warmup_frac: float = 0.10,
    decay_frac: float = 0.20,
) -> Callable[[int], float]:
    """Dispatch by name. Used by the three training stacks."""
    n = name.lower()
    if n == "wsd":
        return build_wsd_schedule(
            total_steps=total_steps, warmup_frac=warmup_frac, decay_frac=decay_frac
        )
    if n == "cosine":
        return build_cosine_schedule(total_steps=total_steps, warmup_frac=warmup_frac)
    if n == "linear":
        return build_linear_schedule(total_steps=total_steps, warmup_frac=warmup_frac)
    raise ValueError(f"Unknown schedule {name!r}; expected wsd/cosine/linear")


# ---------------------------------------------------------------------------
# Width-scaling architecture
# ---------------------------------------------------------------------------
def delphi_arch(
    hidden_size: int,
    *,
    config: Optional[DelphiRecipeConfig] = None,
) -> dict:
    """Return {hidden_size, num_layers, num_heads, intermediate_size, init_std_*}.

    Follows the Delphi width-scaling rule::

        depth = round( H / (64 + 4·log₂H − 9) )
        n_heads = max(1, H // 128)
        intermediate_size = mlp_ratio · H
        init_std_proj = 1/√d_in (per-tensor; consumer applies)
        init_std_emb  = 1/H

    Parameters
    ----------
    hidden_size
        Model width H (must be ≥ 2).
    """
    cfg = config or DelphiRecipeConfig()
    if hidden_size < 2:
        raise ValueError(f"delphi_arch: hidden_size must be >= 2, got {hidden_size}")
    h = float(hidden_size)
    denom = 64.0 + 4.0 * math.log2(h) - 9.0
    depth = max(1, int(round(h / denom)))
    n_heads = max(1, hidden_size // 128)
    # Force head_dim divisibility: hidden_size must be divisible by n_heads.
    while hidden_size % n_heads != 0 and n_heads > 1:
        n_heads -= 1
    return {
        "hidden_size": hidden_size,
        "num_hidden_layers": depth,
        "num_attention_heads": n_heads,
        "num_key_value_heads": max(1, n_heads // 4),  # GQA 4:1 default
        "intermediate_size": cfg.mlp_ratio * hidden_size,
        "init_std_proj_rule": "one_over_sqrt_fan_in",
        "init_std_emb": 1.0 / float(hidden_size),
    }
