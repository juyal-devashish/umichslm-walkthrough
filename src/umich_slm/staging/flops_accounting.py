"""FLOPs and effective-token accounting for scaling-law analysis.

This is the single source of truth for the (N, D, C) point assigned to a
training run in IsoFLOP plots and for the token-horizon term in the
Delphi-style learning-rate schedule.

A "run metadata" dict describes a single SLURM job's compute footprint. It
supports three regimes simultaneously:

  1. Single-stage, non-TST runs::

        {"params": 1.0e8, "tokens": 2.0e9, "arch": "qwen3_like"}

  2. Multi-stage / layer-growth runs::

        {
            "stages": [
                {"params": 5.0e7, "tokens": 1.0e9, "arch": "qwen3_like"},
                {"params": 1.0e8, "tokens": 1.0e9, "arch": "qwen3_like"},
            ],
        }

  3. Token-Superposition Training runs (active phase ingests
     ``bag_size`` tokens per position; recovery is standard)::

        {
            "params": 1.0e8, "arch": "qwen3_like", "seq_len": 4096,
            "tst": {
                "active_steps": 5000, "recovery_steps": 5000,
                "bag_size": 4, "batch_size": 64,
            },
        }

Two functions:

* :func:`total_flops` returns Σ 6·N_i·D_i with per-architecture corrections.
* :func:`effective_tokens` returns the total token count used to position the
  run on the IsoFLOP x-axis and in the LR horizon term.

Per-architecture correction factors are deliberately conservative — they are
multiplicative factors on the canonical ``6·N·D`` estimate that account for
the systematic skew between nominal parameter count and active FLOPs per
token:

* ``mod``: tokens skip a fraction ``1 - mod_capacity`` of the MoD layers'
  forward+backward work, so we multiply by an effective factor.
* ``moe``: only the ``top_k / num_experts`` fraction of expert parameters is
  active per token. We use ``active_params`` for the 6·N·D estimate.
* ``hymba``: SSM cost is roughly linear in state size; we apply a small
  correction (~1.15×) reflecting the extra Mamba2 work per block.

The function never raises on missing optional fields — it returns the
best-effort estimate so partially-populated manifests still plot.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional


def _arch_correction(arch: str, meta: Mapping[str, Any]) -> float:
    """Multiplicative correction on the 6·N·D estimate for the given architecture.

    ``meta`` may carry arch-specific knobs (``mod_capacity``, ``moe_top_k``,
    ``moe_num_experts``, etc.). Missing fields fall back to repo defaults.
    """
    a = (arch or "").lower()
    if "mod" in a and "moda" not in a:
        # MoD: a fraction of layers route only `mod_capacity` of tokens.
        # Assume half the layers are MoD-routed (matches our staged configs).
        cap = float(meta.get("mod_capacity", 0.125))
        routed_layer_frac = float(meta.get("mod_routed_layer_frac", 0.5))
        return routed_layer_frac * cap + (1.0 - routed_layer_frac)
    if "moe" in a:
        top_k = int(meta.get("moe_top_k", 2))
        n_experts = int(meta.get("moe_num_experts", 8))
        if n_experts <= 0:
            return 1.0
        return max(1e-3, top_k / n_experts)
    if "hymba" in a:
        return float(meta.get("hymba_flops_correction", 1.15))
    return 1.0


def _stage_flops(stage: Mapping[str, Any]) -> float:
    n = float(stage["params"])
    d = float(stage["tokens"])
    arch = str(stage.get("arch", ""))
    return 6.0 * n * d * _arch_correction(arch, stage)


def _stages(meta: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    if "stages" in meta and meta["stages"]:
        return meta["stages"]
    return [meta]


def total_flops(run_meta: Mapping[str, Any]) -> float:
    """Total training FLOPs across all stages.

    Returns
    -------
    float
        Sum of ``6·N_i·D_i·correction(arch_i)`` over stages.
    """
    return float(sum(_stage_flops(s) for s in _stages(run_meta)))


def _tst_effective_tokens(tst: Mapping[str, Any], seq_len: int) -> float:
    """Effective token count for a TST-active phase.

    ``effective = active_steps · bag_size · batch · seq + recovery_steps · batch · seq``
    """
    active = int(tst.get("active_steps", 0))
    recovery = int(tst.get("recovery_steps", 0))
    bag = max(1, int(tst.get("bag_size", 1)))
    batch = int(tst.get("batch_size", 1))
    return float((active * bag + recovery) * batch * seq_len)


def effective_tokens(run_meta: Mapping[str, Any], *, raw: bool = False) -> float:
    """Tokens used to position the run on the IsoFLOP x-axis.

    Parameters
    ----------
    run_meta
        Run metadata dict (see module docstring).
    raw
        If True, ignore the TST bag-size multiplier (i.e. count physical
        token positions seen, not the superposed bag). Used for the
        "raw-tokens" supplementary plot.
    """
    total = 0.0
    for s in _stages(run_meta):
        if "tst" in s and s["tst"] and not raw:
            seq_len = int(s.get("seq_len", run_meta.get("seq_len", 4096)))
            total += _tst_effective_tokens(s["tst"], seq_len)
        elif "tst" in s and s["tst"] and raw:
            tst = s["tst"]
            seq_len = int(s.get("seq_len", run_meta.get("seq_len", 4096)))
            steps = int(tst.get("active_steps", 0)) + int(tst.get("recovery_steps", 0))
            batch = int(tst.get("batch_size", 1))
            total += float(steps * batch * seq_len)
        else:
            total += float(s["tokens"])
    return total


def final_params(run_meta: Mapping[str, Any]) -> int:
    """Parameter count assigned to the run as a single IsoFLOP point.

    For multi-stage runs this is the *final* stage's parameter count, which
    matches what the headline IsoFLOP plot wants. Use :func:`stage_points`
    if you need per-stage (N_i, D_i) instead.
    """
    stages = list(_stages(run_meta))
    return int(stages[-1]["params"])


def stage_points(run_meta: Mapping[str, Any]) -> list[Dict[str, float]]:
    """Per-stage (N, D, partial-C) tuples for the supplementary staging plot."""
    out: list[Dict[str, float]] = []
    for s in _stages(run_meta):
        out.append(
            {
                "params": float(s["params"]),
                "tokens": float(s.get("tokens", 0.0)),
                "flops": _stage_flops(s),
                "arch": str(s.get("arch", "")),
            }
        )
    return out


def chinchilla_optimal_tokens(flops: float, params: int) -> float:
    """Solve D from C = 6·N·D — used to derive token budgets from FLOPs."""
    return flops / (6.0 * float(params))


def chinchilla_optimal_params(flops: float) -> float:
    """N* = √(C/6) under the equal-scaling Chinchilla heuristic."""
    return (flops / 6.0) ** 0.5
