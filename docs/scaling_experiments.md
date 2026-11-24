# Scaling Experiments

Methodology, sweep design, and plot reference for the **Delphi-style**
scaling-law experiments on UMichSLM.

The previous Chinchilla-style sweep (preserved in `scripts/scaling_laws.py
chinchilla` + the legacy code path in `scripts/plot_isoflop.py`) suffered
from monotone upward IsoFLOP curves — every model was under-trained, so no
parabola minimum was visible and the held-out forecast was meaningless.
The Delphi recipe (openathena.ai/blog/delphi) addresses both: a
token-horizon-aware LR + Hyperball optimizer + WSD schedule produces clean
IsoFLOP parabolas, and a parametric ``L(N, D) = E + A/N^α + B/D^β`` fit
on the per-budget optima lets us extrapolate to a held-out budget and
quantify residual.

See `docs/scaling_recipe.md` for the recipe formulas and configurable
knobs. This document is about *how* we use the recipe to make decisions.

## Two-pass workflow

1. **Screening pass** — every architecture family at every size, single seed:

       python scripts/scaling_laws.py delphi_sweep \
           --output_dir generated-slurm/delphi_screen \
           --slurm_config configs/slurm/delphi_sweep_greatlakes.yaml \
           --pass screen

   Default: 4 sizes (`10m, 25m, 50m, 100m`) × 4 budgets ×
   `{qwen3_like, qwen3_5_like, mod, hymba_like, delphi}`. Staged variants
   are emitted for archs listed in `--stage_archs` (default
   `delphi,qwen3_like`) at 50m/100m only.

2. **Ranking and selection** — after pass 1 finishes:

       python scripts/scaling_fit.py \
           --manifest generated-slurm/delphi_screen/sweep_manifest.json \
           --results_dir /path/to/data \
           --output_dir figures/delphi_screen \
           --holdout_budget 1e19 \
           --rank

   The printed table orders architectures by predicted held-out loss
   (lower = better extrapolation). Use it to pick the ≈2–4 architectures
   that advance to pass 2.

3. **Held-out pass** — only the selected subset, 3 seeds per point:

       python scripts/scaling_laws.py delphi_sweep \
           --output_dir generated-slurm/delphi_holdout \
           --slurm_config configs/slurm/delphi_sweep_greatlakes.yaml \
           --pass holdout \
           --archs delphi,qwen3_like \
           --seeds 0,1,2

   After all runs complete, re-run `scaling_fit.py` against the merged
   manifest (the writer appends across passes); held-out residuals are
   then populated in the ranking table.

## Cross-architecture comparison

Every plot supports overlay by architecture:

* `figures/.../isoflop_overlay.png` — one parabola per (arch, budget)
  with star markers at the parabolic minimum.
* `figures/.../exponents_bar.png` — fitted (α, β) per arch side by side.
* `figures/.../holdout_residuals.png` — residual % per arch at the
  held-out budget.
* `figures/.../overtraining_cost.png` — extra-compute multiplier to match
  optimal loss at k× overtrained tokens, per arch.

The ranking table prints the same data in tabular form, useful for PR
descriptions and for arguing for or against advancing an architecture.

## Staging and TST conventions

Layer-stacking runs (`stages` in the manifest) and Token-Superposition
Training runs (`tst` in the manifest) appear *twice* in the figures:

* In the **headline** IsoFLOP plot, each run is a single (N, D, C) point.
  N is the final-stage parameter count; D and C are the sum across stages.
  Effective tokens for TST use the bag-size multiplier.
* In **supplementary** plots (run `scripts/plot_staging_tst.py`):
  * `staging_trajectory.png` — arrows in (N, D) space showing each stage.
  * `staging_cumulative_flops.png` — loss vs cumulative FLOPs; dotted
    verticals mark stage boundaries.
  * `tst_dual_axis.png` — loss vs raw tokens (bottom axis) and effective
    tokens (top axis).

All accounting flows through `src/umich_slm/staging/flops_accounting.py`
— do not reimplement these formulas in the plotters.

## Downstream-benchmark scaling

Hard scores at 50M–100M are at chance on most benchmarks. We use a *soft*
metric (log-prob of correct option, or bits-per-byte on a reference
completion) and a sigmoid fit on an external reference pool
(`evals/reference_models.json`) to project hard scores at the held-out
budget:

    python scripts/plot_benchmark_scaling.py \
        --eval_log path/to/eval_results.jsonl \
        --reference evals/reference_models.json \
        --output_dir figures/benchmarks \
        --holdout_flops 1e19

See `docs/evaluation.md` for the soft-metric definitions and the agentic
benchmark MC adapter (`evals/agentbench_mc.py`).

## Pass-1 ranking criterion

After pass 1, the ranking table shows per-arch:

* Fitted `(E, A, B, α, β)` and 95 % bootstrap CIs (resampling screening
  optima).
* Predicted compute-optimal (N\*, D\*) at the held-out budget.
* Predicted held-out loss `L*`.

The selection itself is not automated. A reasonable rule of thumb: pick
the architectures whose `L*` is within ~5 % of the best, prioritising
those with larger `β` (token-efficiency dominates at our scale) and
smaller bootstrap CIs (extrapolation is more trustworthy).

## Pre-existing Chinchilla path

The legacy path is preserved for reproducing the old sweep:

    python scripts/scaling_laws.py chinchilla --output_dir ./chinchilla_experiments
    python scripts/plot_isoflop.py --result_dirs ./chinchilla_experiments

This path uses `compute_optimal_lr` (μP-style) and the hard-coded
Chinchilla coefficients in `plot_isoflop.py`. It does not require the
Delphi recipe.

## Smoke verification

A minimal end-to-end smoke runs in ~30 s on CPU:

    python scripts/smoke_delphi_sweep.py --output_dir /tmp/delphi_smoke

It exercises AdamH (norm preservation), WSD (end-LR = 0), FLOPs accounting,
manifest round-trip, and the `scaling_fit.py` pipeline on synthetic
training logs. It prints `PASS` or `FAIL`.

## Troubleshooting

* **All IsoFLOP curves slope upward** — sizes are too large for the
  budget. Check `tokens_per_param = D / N` — Chinchilla suggests ≥ 20.
* **Held-out residual > 10 %** — fit set probably has only one parabola
  optimum visible. Add more budgets at the small end.
* **`scaling_fit.py` reports "skipped: only N screening points"** — need
  ≥ 5 points per arch for the power-law fit. Add more sizes or budgets.
* **AdamH norm drift > 1 %** — a tensor that should be Hyperball-constrained
  is in the wrong group. Inspect `split_param_groups` exclude substrings.
* **β₂ clip clamps to 0.9 immediately** — global batch is huge relative
  to `B₀ = 64`. Either lower the batch or raise `B₀` (`--lr_batch_ref`).
