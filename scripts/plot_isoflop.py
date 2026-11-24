#!/usr/bin/env python3
"""
IsoFLOP curve plotter for UMich SLM scaling experiments.
Reads trainer_state.json from all experiment folders, extracts final
training loss, computes FLOPs, and generates 3 plots:
  1. IsoFLOP curves with Chinchilla theoretical overlay
  2. Loss vs token budget per architecture
  3. Loss vs model size per token budget

Usage:
    python scripts/plot_isoflop.py

    # with custom paths
    python scripts/plot_isoflop.py \
        --result_dirs /path/to/user /path/to/user/results \
        --output_dir /path/to/output
"""

import argparse
import json
import os
import math
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


#----- CONFIG -----

# Default paths for Great Lakes - override via --result_dirs and --output_dir
DEFAULT_RESULT_DIRS = [
    "/path/to/data",
    "/path/to/data",
]

DEFAULT_OUTPUT_DIR = "/path/to/data"

# Backbone parameter counts from architecture_presets.md
PARAM_COUNTS = {
    "qwen3_5_like_tiny": 4.9e6,
    "qwen3_5_like_xsmall": 19.9e6,
    "qwen3_5_like_small": 99.8e6,
    "qwen3_like_tiny": 5.0e6,
    "qwen3_like_xsmall": 19.9e6,
    "qwen3_like_small": 100.7e6,
    "mod_tiny": 30e6,
    "mod_xsmall": 78e6,
    "mod_small": 252e6,
    "hymba_like_tiny": 26e6,
    "hymba_like_xsmall": 62e6,
}

ARCH_LABELS = {
    "qwen3_5_like_tiny": "Qwen3.5 Tiny (5M)",
    "qwen3_5_like_xsmall": "Qwen3.5 XSmall (20M)",
    "qwen3_5_like_small": "Qwen3.5 Small (100M)",
    "qwen3_like_tiny": "Qwen3 Tiny (5M)",
    "qwen3_like_xsmall": "Qwen3 XSmall (20M)",
    "qwen3_like_small": "Qwen3 Small (101M)",
    "mod_tiny": "MoD Tiny (30M)",
    "mod_xsmall": "MoD XSmall (78M)",
    "mod_small": "MoD Small (252M)",
    "hymba_like_tiny": "Hymba Tiny (26M)",
    "hymba_like_xsmall": "Hymba XSmall (62M)",
}

ARCH_COLORS = {
    "qwen3_5_like_tiny": "#1f77b4",
    "qwen3_5_like_xsmall": "#4fa8d5",
    "qwen3_5_like_small": "#aed6ef",
    "qwen3_like_tiny": "#d62728",
    "qwen3_like_xsmall": "#e87272",
    "qwen3_like_small": "#f5b7b7",
    "mod_tiny": "#1a6e1a",
    "mod_xsmall": "#2ca02c",
    "mod_small": "#6dcf6d",
    "hymba_like_tiny": "#7b2d8b",
    "hymba_like_xsmall": "#9467bd",
}


#----- DATA LOADING -----

def estimate_flops(params, tokens):
    # Chinchilla approximation: C = 6 * N * D
    return 6 * params * tokens


def parse_folder_name(folder_name):
    # Extract arch name and token count from e.g. mod_xsmall_78M_106Mtok
    name = folder_name.strip("/")
    last = name.split("_")[-1]
    if last.endswith("Mtok"):
        tokens = float(last.replace("Mtok", "")) * 1e6
    elif last.endswith("Btok"):
        tokens = float(last.replace("Btok", "")) * 1e9
    else:
        return None, None
    # match longest arch name first to avoid partial matches
    arch = None
    for candidate in sorted(PARAM_COUNTS.keys(), key=len, reverse=True):
        if name.startswith(candidate):
            arch = candidate
            break
    return arch, tokens


def find_trainer_state(exp_dir):
    candidates = []
    for pattern in [os.path.join(exp_dir, "*", "trainer_state.json"),
                    os.path.join(exp_dir, "*", "*", "trainer_state.json")]:
        for path in glob.glob(pattern):
            # skip checkpoint subfolders
            if any(p.startswith("checkpoint-") for p in path.split(os.sep)):
                continue
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=os.path.getmtime, reverse=True)[0]


def get_final_loss(path):
    try:
        with open(path) as f:
            data = json.load(f)
        entries = [e for e in data.get("log_history", []) if "loss" in e]
        if not entries:
            return None
        loss = entries[-1]["loss"]
        # filter crashed runs
        if loss == 0 or math.isnan(loss) or math.isinf(loss) or loss < 0.5:
            return None
        return loss
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return None


def collect_results(result_dirs):
    results = []
    for base_dir in result_dirs:
        if not os.path.isdir(base_dir):
            print(f"Skipping missing dir: {base_dir}")
            continue
        for folder in sorted(os.listdir(base_dir)):
            path = os.path.join(base_dir, folder)
            if not os.path.isdir(path):
                continue
            arch, tokens = parse_folder_name(folder)
            if arch is None or arch not in PARAM_COUNTS:
                continue
            ts = find_trainer_state(path)
            if ts is None:
                continue
            loss = get_final_loss(ts)
            if loss is None:
                continue
            params = PARAM_COUNTS[arch]
            results.append({
                "arch": arch,
                "tokens": tokens,
                "params": params,
                "loss": loss,
                "flops": estimate_flops(params, tokens),
            })
    return results


#----- CHINCHILLA THEORY -----

def chinchilla_optimal(C, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    # Returns (optimal_params, optimal_tokens, expected_loss) for compute budget C
    N_opt = ((alpha * A) / (beta * B)) ** (1 / (alpha + beta)) * (C / 6) ** (beta / (alpha + beta))
    D_opt = C / (6 * N_opt)
    loss = E + A / (N_opt ** alpha) + B / (D_opt ** beta)
    return N_opt, D_opt, loss


#----- GRAPH 1: ISOFLOP CURVES -----

def plot_isoflop(results, ax):
    families = {
        "MoD": {
            "archs": ["mod_xsmall", "mod_small"],
            "colors": ["#2ca02c", "#1a6e1a", "#006400", "#004000"],
        },
        "Hymba": {
            "archs": ["hymba_like_tiny", "hymba_like_xsmall"],
            "colors": ["#9467bd", "#7b2d8b", "#4b0082", "#2d004b"],
        },
    }

    def flop_bucket(f):
        # group runs by rounding log10(FLOPs) to nearest 0.5
        return round(math.log10(f) * 2) / 2

    for family_name, info in families.items():
        family_runs = [r for r in results if r["arch"] in info["archs"]]
        if not family_runs:
            continue
        groups = {}
        for r in family_runs:
            groups.setdefault(flop_bucket(r["flops"]), []).append(r)
        for idx, bucket in enumerate(sorted(groups.keys())):
            group = sorted(groups[bucket], key=lambda x: x["params"])
            if len(group) < 2:
                continue
            xs = [r["params"] / 1e6 for r in group]
            ys = [r["loss"] for r in group]
            color = info["colors"][idx % len(info["colors"])]
            ax.plot(xs, ys, "o-", color=color, linewidth=2.0, markersize=8,
                    label=f"{family_name} C={10**bucket:.0e}", zorder=3)

    # Chinchilla theoretical curve
    budgets = np.logspace(14, 18, 200)
    opt_params = [chinchilla_optimal(C)[0] / 1e6 for C in budgets]
    opt_losses = [chinchilla_optimal(C)[2] for C in budgets]
    ax.plot(opt_params, opt_losses, "k--", linewidth=2.0,
            label="Chinchilla optimal (theoretical)", alpha=0.7, zorder=5)

    # stars at our actual compute budgets on the optimal curve
    actual_budgets = sorted(set(
        round(r["flops"] / 1e15, 1) * 1e15
        for r in results
        if r["arch"] in ["mod_xsmall", "mod_small", "hymba_like_tiny", "hymba_like_xsmall"]
    ))
    for C in actual_budgets:
        N_opt, _, loss = chinchilla_optimal(C)
        ax.scatter(N_opt / 1e6, loss, marker="*", s=200, color="black", zorder=6)

    ax.set_xlabel("Model Parameters (M)", fontsize=11)
    ax.set_ylabel("Final Training Loss", fontsize=11)
    ax.set_title("IsoFLOP Curves\n(dashed = Chinchilla optimal, stars = optimal points for our budgets)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)


#----- GRAPH 2: LOSS VS TOKENS -----

def plot_loss_vs_tokens(results, ax):
    arch_groups = {}
    for r in results:
        arch_groups.setdefault(r["arch"], []).append(r)

    for arch, runs in sorted(arch_groups.items()):
        runs.sort(key=lambda x: x["tokens"])
        xs = [r["tokens"] / 1e6 for r in runs]
        ys = [r["loss"] for r in runs]
        color = ARCH_COLORS.get(arch, "gray")
        label = ARCH_LABELS.get(arch, arch)
        if len(runs) < 2:
            ax.scatter(xs, ys, color=color, s=80, zorder=5, label=label)
        else:
            ax.plot(xs, ys, "o-", color=color, label=label, linewidth=2.0, markersize=7)

    ax.set_xlabel("Training Tokens (M)", fontsize=11)
    ax.set_ylabel("Final Training Loss", fontsize=11)
    ax.set_title("Loss vs Token Budget\n(more data = lower loss for all architectures)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")


#----- GRAPH 3: LOSS VS PARAMS -----

def plot_loss_vs_params(results, ax):
    def tok_label(t):
        for threshold, label in [
            (15e6, "~13M"), (25e6, "~20M"), (50e6, "~40M"),
            (80e6, "~66M"), (120e6, "~106M"), (175e6, "~135M"),
            (250e6, "~213M"), (310e6, "~270M"), (360e6, "~320M"),
            (500e6, "~331M"), (600e6, "~552M"), (650e6, "~640M"),
            (700e6, "~662M"), (2000e6, "~1B"),
        ]:
            if t < threshold:
                return label
        return f"{t/1e6:.0f}M"

    tok_groups = {}
    for r in results:
        tok_groups.setdefault(tok_label(r["tokens"]), []).append(r)

    def sort_key(label):
        try:
            return float(label.replace("~", "").replace("M", "").replace("B", "000"))
        except:
            return 0

    sorted_labels = sorted(tok_groups.keys(), key=sort_key)
    colors = cm.viridis(np.linspace(0.05, 0.95, max(len(sorted_labels), 1)))

    for idx, label in enumerate(sorted_labels):
        group = sorted(tok_groups[label], key=lambda x: x["params"])
        xs = [r["params"] / 1e6 for r in group]
        ys = [r["loss"] for r in group]
        if len(group) < 2:
            ax.scatter(xs, ys, color=colors[idx], s=80, zorder=5, label=f"{label} tokens")
        else:
            ax.plot(xs, ys, "o-", color=colors[idx], label=f"{label} tokens",
                    linewidth=2.0, markersize=7)

    ax.set_xlabel("Model Parameters (M)", fontsize=11)
    ax.set_ylabel("Final Training Loss", fontsize=11)
    ax.set_title("Loss vs Model Size\n(larger models need more tokens to beat smaller ones)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", title="Token Budget")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")


#----- SAVE PLOTS -----

def make_plots(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not results:
        print("No results to plot.")
        return

    plots = [
        (plot_isoflop, "isoflop_curves.png", "IsoFLOP Curves"),
        (plot_loss_vs_tokens, "loss_vs_tokens.png", "Loss vs Tokens"),
        (plot_loss_vs_params, "loss_vs_params.png", "Loss vs Model Size"),
    ]

    for fn, filename, title in plots:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.suptitle(f"UMich SLM Scaling - {title}", fontsize=13, fontweight="bold")
        fn(results, ax)
        plt.tight_layout()
        out = os.path.join(output_dir, filename)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")

    # combined figure with all 3 side by side
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle("UMich SLM Scaling Experiments - Chinchilla Analysis",
                 fontsize=14, fontweight="bold")
    plot_isoflop(results, axes[0])
    plot_loss_vs_tokens(results, axes[1])
    plot_loss_vs_params(results, axes[2])
    plt.tight_layout()
    out = os.path.join(output_dir, "all_scaling_graphs.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


#----- MAIN -----

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate IsoFLOP scaling curves")
    parser.add_argument(
        "--result_dirs",
        nargs="+",
        default=DEFAULT_RESULT_DIRS,
        help="Directories containing experiment result folders",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output plots",
    )
    args = parser.parse_args()

    print("Result dirs:")
    for d in args.result_dirs:
        print(f"  {d}")
    print(f"Output dir: {args.output_dir}\n")

    print("Collecting results...")
    results = collect_results(args.result_dirs)
    print(f"Found {len(results)} valid runs\n")

    print(f"{'Architecture':<25} {'Tokens':>10} {'Params':>10} {'Loss':>8} {'FLOPs':>12}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: (x["arch"], x["tokens"])):
        print(f"{r['arch']:<25} {r['tokens']/1e6:>9.0f}M "
              f"{r['params']/1e6:>9.1f}M {r['loss']:>8.4f} {r['flops']:>12.2e}")

    print("\nGenerating plots...")
    make_plots(results, args.output_dir)
    print(f"\nDone. Files saved to: {args.output_dir}")
