# Scaling Experiments

This document summarises the scaling law experiments run by the scaling team,
the analysis methodology, key findings, and recommendations for future runs.

## Overview

We trained models across four architecture families at multiple token budgets
to study how loss scales with compute, following the Chinchilla (2022) methodology.

## Architectures Tested

| Architecture | Type | Sizes |
|---|---|---|
| qwen3_like | Pure attention (GQA) | tiny (5M), xsmall (20M), small (101M) |
| qwen3_5_like | Hybrid DeltaNet+Attention | tiny (5M), xsmall (20M), small (100M) |
| MoD | Mixture of Depths | tiny (30M), xsmall (78M), small (252M) |
| Hymba | Hybrid SSM+Attention | tiny (26M), xsmall (62M) |

## Experiment Results

IsoFLOP analysis used 20 completed runs from MoD and Hymba experiments.
qwen runs were excluded as training logs were not saved alongside model weights.

Token budgets covered: 11M, 13M, 21M, 27M, 33M, 55M, 64M, 66M, 106M,
135M, 213M, 270M, 320M, 331M, 552M, 640M, 662M, and 1000M tokens.

## Key Findings

All IsoFLOP lines slope upward. For every compute budget tested, the smaller
model outperformed the larger one. Example: MoD small (252M params, 33M tokens)
and MoD xsmall (78M params, 106M tokens) cost identical FLOPs (~5e16) but
achieved loss 7.34 vs 5.44 respectively.

Our data points sit to the right of the Chinchilla optimal curve, meaning we
over-allocated compute to model size and under-allocated to data. Larger models
did not receive enough tokens to justify their parameter count. Chinchilla
recommends roughly 20 tokens per parameter at minimum.

| Model | Params | Chinchilla min tokens | Max tokens used |
|---|---|---|---|
| MoD xsmall | 78M | 1,560M | 1,000M |
| MoD small | 252M | 5,040M | 662M |

## How to Generate Plots

Run the IsoFLOP plotter from the repo root:

    python scripts/plot_isoflop.py

    python scripts/plot_isoflop.py \
        --result_dirs /path/to/results_a /path/to/results_b \
        --output_dir /path/to/output

Outputs 4 files: isoflop_curves.png, loss_vs_tokens.png,
loss_vs_params.png, and all_scaling_graphs.png.

## Recommendations for Future Runs

To generate proper U-shaped IsoFLOP curves, future experiments should fix
the compute budget first and derive token counts per model size:

    tokens = C / (6 * params)

Example for C = 1e17 FLOPs:

| Model | Params | Tokens |
|---|---|---|
| MoD tiny | 30M | 556M |
| MoD xsmall | 78M | 214M |
| MoD small | 252M | 66M |

All three cost exactly 1e17 FLOPs. Plotting loss vs params across these
runs should produce a U-shape with the optimal model size at the minimum.
