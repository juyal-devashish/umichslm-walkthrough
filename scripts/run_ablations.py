#!/usr/bin/env python3
"""Run ablation experiments for UMichSLM pretraining.

This script runs multiple pretraining experiments with different feature
configurations (MoD, RoPE/PoPE, MTP, etc.) and logs results to wandb.

Example usage:
    # Run all ablations (dry run first to see what will run)
    python scripts/run_ablations.py --dry_run --wandb_project umichslm-ablations

    # Run all ablations
    python scripts/run_ablations.py --wandb_project umichslm-ablations

    # Run specific ablations
    python scripts/run_ablations.py --ablations position_embedding mod --wandb_project umichslm

    # Run with custom settings
    python scripts/run_ablations.py --size tiny --max_steps 1000 --wandb_project test
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str
    base_args: Dict[str, Any] = field(default_factory=dict)
    variations: List[Dict[str, Any]] = field(default_factory=list)


# Define ablation experiments
ABLATIONS: Dict[str, AblationConfig] = {
    "baseline": AblationConfig(
        name="baseline",
        description="Baseline model without advanced features",
        variations=[
            {"_suffix": "baseline"},  # Just the base model
        ],
    ),
    "position_embedding": AblationConfig(
        name="position_embedding",
        description="Compare RoPE vs PoPE position embeddings",
        variations=[
            {"position_embedding_type": "rope", "_suffix": "rope"},
            {"position_embedding_type": "pope", "_suffix": "pope"},
        ],
    ),
    "mod": AblationConfig(
        name="mod",
        description="Mixture of Depths ablation",
        variations=[
            {"use_mod": False, "_suffix": "no_mod"},
            {"use_mod": True, "mod_capacity": 0.125, "_suffix": "mod_12.5"},
            {"use_mod": True, "mod_capacity": 0.25, "_suffix": "mod_25"},
            {"use_mod": True, "mod_capacity": 0.5, "_suffix": "mod_50"},
        ],
    ),
    "mod_layers": AblationConfig(
        name="mod_layers",
        description="MoD layer frequency ablation",
        base_args={"use_mod": True, "mod_capacity": 0.125},
        variations=[
            {"mod_every_n_layers": 1, "_suffix": "every1"},  # Every layer
            {"mod_every_n_layers": 2, "_suffix": "every2"},  # Alternating
            {"mod_every_n_layers": 3, "_suffix": "every3"},
            {"mod_every_n_layers": 4, "_suffix": "every4"},
        ],
    ),
    "mtp": AblationConfig(
        name="mtp",
        description="Multi-Token Prediction ablation",
        variations=[
            {"use_mtp": False, "_suffix": "no_mtp"},
            {"use_mtp": True, "mtp_num_heads": 1, "_suffix": "mtp_1"},
            {"use_mtp": True, "mtp_num_heads": 2, "_suffix": "mtp_2"},
            {"use_mtp": True, "mtp_num_heads": 3, "_suffix": "mtp_3"},
        ],
    ),
    "mtp_weight": AblationConfig(
        name="mtp_weight",
        description="MTP loss weight ablation",
        base_args={"use_mtp": True, "mtp_num_heads": 1},
        variations=[
            {"mtp_loss_weight": 0.1, "_suffix": "weight_0.1"},
            {"mtp_loss_weight": 0.3, "_suffix": "weight_0.3"},
            {"mtp_loss_weight": 0.5, "_suffix": "weight_0.5"},
            {"mtp_loss_weight": 1.0, "_suffix": "weight_1.0"},
        ],
    ),
    "moe": AblationConfig(
        name="moe",
        description="Mixture of Experts ablation",
        variations=[
            {"use_moe": False, "_suffix": "dense"},
            {"use_moe": True, "num_experts": 4, "num_experts_per_tok": 1, "_suffix": "moe_4x1"},
            {"use_moe": True, "num_experts": 8, "num_experts_per_tok": 2, "_suffix": "moe_8x2"},
            {"use_moe": True, "num_experts": 16, "num_experts_per_tok": 2, "_suffix": "moe_16x2"},
        ],
    ),
    "sliding_window": AblationConfig(
        name="sliding_window",
        description="Sliding Window Attention ablation",
        variations=[
            {"sliding_window": None, "_suffix": "full_attn"},
            {"sliding_window": 512, "_suffix": "sw_512"},
            {"sliding_window": 1024, "_suffix": "sw_1024"},
            {"sliding_window": 2048, "_suffix": "sw_2048"},
        ],
    ),
    "combined": AblationConfig(
        name="combined",
        description="Combined features (best of each)",
        variations=[
            # Baseline
            {"_suffix": "baseline"},
            # MoD only
            {"use_mod": True, "mod_capacity": 0.125, "_suffix": "mod"},
            # MTP only
            {"use_mtp": True, "mtp_num_heads": 1, "_suffix": "mtp"},
            # MoD + MTP
            {"use_mod": True, "mod_capacity": 0.125,
             "use_mtp": True, "mtp_num_heads": 1, "_suffix": "mod_mtp"},
            # MoD + MTP + PoPE
            {"use_mod": True, "mod_capacity": 0.125,
             "use_mtp": True, "mtp_num_heads": 1,
             "position_embedding_type": "pope", "_suffix": "mod_mtp_pope"},
        ],
    ),
    "full_sweep": AblationConfig(
        name="full_sweep",
        description="Full factorial sweep of key features",
        variations=[],  # Will be generated dynamically
    ),
}


def generate_full_sweep() -> List[Dict[str, Any]]:
    """Generate full factorial sweep of key features."""
    variations = []

    # Define feature options
    position_types = ["rope", "pope"]
    mod_options = [
        {"use_mod": False},
        {"use_mod": True, "mod_capacity": 0.125},
    ]
    mtp_options = [
        {"use_mtp": False},
        {"use_mtp": True, "mtp_num_heads": 1},
    ]

    # Generate combinations
    for pos_type, mod_opt, mtp_opt in itertools.product(
        position_types, mod_options, mtp_options
    ):
        config = {"position_embedding_type": pos_type}
        config.update(mod_opt)
        config.update(mtp_opt)

        # Generate suffix
        parts = [pos_type]
        if config.get("use_mod"):
            parts.append("mod")
        if config.get("use_mtp"):
            parts.append("mtp")
        config["_suffix"] = "_".join(parts)

        variations.append(config)

    return variations


# Generate full sweep variations
ABLATIONS["full_sweep"].variations = generate_full_sweep()


def build_pretrain_command(
    variation: Dict[str, Any],
    ablation_name: str,
    base_args: Dict[str, Any],
    common_args: argparse.Namespace,
) -> List[str]:
    """Build the pretrain.py command for a single variation."""
    # Remove internal suffix key
    suffix = variation.pop("_suffix", "")
    variation = {k: v for k, v in variation.items() if not k.startswith("_")}

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "pretrain.py"),
    ]

    # Add common arguments
    cmd.extend(["--size", common_args.size])
    cmd.extend(["--dataset", common_args.dataset])
    cmd.extend(["--output_dir", common_args.output_dir])
    cmd.extend(["--max_steps", str(common_args.max_steps)])
    cmd.extend(["--batch_size", str(common_args.batch_size)])
    cmd.extend(["--gradient_accumulation_steps", str(common_args.gradient_accumulation_steps)])
    cmd.extend(["--learning_rate", str(common_args.learning_rate)])
    cmd.extend(["--logging_steps", str(common_args.logging_steps)])
    cmd.extend(["--save_steps", str(common_args.save_steps)])
    cmd.extend(["--seed", str(common_args.seed)])
    if common_args.deterministic:
        cmd.append("--deterministic")

    if common_args.wandb_project:
        cmd.extend(["--wandb_project", common_args.wandb_project])
    if common_args.wandb_entity:
        cmd.extend(["--wandb_entity", common_args.wandb_entity])

    # Generate run name with date
    date_str = datetime.now().strftime("%Y%m%d")
    run_name = f"{ablation_name}_{suffix}_{common_args.size}_{date_str}"
    cmd.extend(["--run_name", run_name])

    # Add base args for this ablation
    for key, value in base_args.items():
        if key.startswith("_"):
            continue
        cmd.extend([f"--{key}", str(value)])

    # Add variation-specific args
    for key, value in variation.items():
        if key.startswith("_"):
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif value is not None:
            cmd.extend([f"--{key}", str(value)])

    return cmd


def run_experiment(cmd: List[str], dry_run: bool = False) -> int:
    """Run a single experiment."""
    cmd_str = " ".join(cmd)
    print(f"\n{'=' * 60}")
    print(f"Running: {cmd_str}")
    print("=" * 60)

    if dry_run:
        print("[DRY RUN] Would execute above command")
        return 0

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for UMichSLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Ablation selection
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["combined"],
        choices=list(ABLATIONS.keys()) + ["all"],
        help="Which ablation(s) to run",
    )
    parser.add_argument(
        "--list_ablations",
        action="store_true",
        help="List available ablations and exit",
    )

    # Common training settings
    parser.add_argument("--size", default="small", help="Model size")
    parser.add_argument("--dataset", default="smollm_cosmopedia", help="Dataset name")
    parser.add_argument("--output_dir", default="./outputs/ablations", help="Output directory")
    parser.add_argument("--max_steps", type=int, default=10000, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging interval")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Enable fully deterministic mode (may slow training by 5-15%%)")

    # Wandb
    parser.add_argument("--wandb_project", default=None, help="Wandb project name")
    parser.add_argument("--wandb_entity", default=None, help="Wandb entity")

    # Execution options
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--sequential", action="store_true", help="Run experiments sequentially (default)")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop if any experiment fails")

    args = parser.parse_args()

    # Set random seeds for reproducibility
    from umich_slm.utils import set_seed
    set_seed(args.seed, deterministic=args.deterministic)

    # List ablations if requested
    if args.list_ablations:
        print("Available ablations:")
        print("-" * 60)
        for name, config in ABLATIONS.items():
            print(f"  {name:20s} - {config.description}")
            print(f"                       {len(config.variations)} variations")
        sys.exit(0)

    # Determine which ablations to run
    if "all" in args.ablations:
        ablation_names = list(ABLATIONS.keys())
    else:
        ablation_names = args.ablations

    # Validate ablation names
    for name in ablation_names:
        if name not in ABLATIONS:
            print(f"Error: Unknown ablation '{name}'")
            sys.exit(1)

    # Count total experiments
    total_experiments = sum(len(ABLATIONS[name].variations) for name in ablation_names)
    print(f"\n{'=' * 60}")
    print(f"UMichSLM Ablation Study")
    print(f"{'=' * 60}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ablations: {ablation_names}")
    print(f"Total experiments: {total_experiments}")
    print(f"Model size: {args.size}")
    print(f"Max steps per experiment: {args.max_steps}")
    print(f"Output directory: {args.output_dir}")
    if args.wandb_project:
        print(f"Wandb project: {args.wandb_project}")
    print(f"{'=' * 60}")

    # Collect all experiments
    experiments = []
    for ablation_name in ablation_names:
        ablation = ABLATIONS[ablation_name]
        for variation in ablation.variations:
            # Make a copy to avoid modifying the original
            var_copy = variation.copy()
            cmd = build_pretrain_command(
                var_copy,
                ablation_name,
                ablation.base_args,
                args,
            )
            experiments.append({
                "ablation": ablation_name,
                "variation": variation.get("_suffix", "default"),
                "command": cmd,
            })

    # Print experiment summary
    print("\nExperiments to run:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i:3d}. {exp['ablation']}/{exp['variation']}")

    # Run experiments
    results = []
    failed = []

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Running: {exp['ablation']}/{exp['variation']}")

        return_code = run_experiment(exp["command"], dry_run=args.dry_run)

        result = {
            "ablation": exp["ablation"],
            "variation": exp["variation"],
            "return_code": return_code,
            "success": return_code == 0,
        }
        results.append(result)

        if return_code != 0:
            failed.append(exp)
            if args.stop_on_error:
                print(f"\nError: Experiment failed. Stopping (--stop_on_error).")
                break

    # Summary
    print(f"\n{'=' * 60}")
    print("Ablation Study Complete")
    print(f"{'=' * 60}")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")

    if failed:
        print("\nFailed experiments:")
        for exp in failed:
            print(f"  - {exp['ablation']}/{exp['variation']}")

    # Save results
    if not args.dry_run:
        results_path = Path(args.output_dir) / f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "args": vars(args),
                "experiments": results,
            }, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    # Exit with error if any failed
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
