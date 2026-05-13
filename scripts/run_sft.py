#!/usr/bin/env python3
"""Launch SFT training with a resolved LLaMA Factory config.

Run from the repository root.

Examples:
    python scripts/run_sft.py --dataset smoltalk
    python scripts/run_sft.py --dataset openmath2 --max_samples 200000
    python scripts/run_sft.py --dataset umichslm_sft_train --dry_run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - used in minimal local environments
    yaml = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from slurm_common import load_yaml_config, shell_join  # noqa: E402


# Map each dataset to its appropriate base config.
DATASET_CONFIG_MAP = {
    "smoltalk": "configs/llama_factory/umichslm_sft.yaml",
    "openmath2": "configs/llama_factory/umichslm_reasoning_warmup.yaml",
    "umichslm_sft_train": "configs/llama_factory/umichslm_sft.yaml",
}

SUPPORTED_DATASETS = sorted(DATASET_CONFIG_MAP)


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    """Write YAML, falling back to a small dumper when PyYAML is unavailable."""
    if yaml is not None:
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return

    def render_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        return str(value)

    lines = [f"{key}: {render_value(value)}" for key, value in data.items()]
    path.write_text("\n".join(lines) + "\n")


def resolve_sft_config(
    dataset: str,
    base_config: str | None = None,
    max_samples: int | None = None,
    output_suffix: str | None = None,
    output_dir: str | None = None,
    logging_dir: str | None = None,
    model_name_or_path: str | None = None,
    eval_dataset: str | None = None,
    learning_rate: float | None = None,
    num_train_epochs: float | None = None,
    per_device_train_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    save_steps: int | None = None,
    eval_steps: int | None = None,
    logging_steps: int | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Resolve a concrete LLaMA Factory SFT config."""
    if dataset not in DATASET_CONFIG_MAP:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choices: {', '.join(SUPPORTED_DATASETS)}")

    config_path = base_config or DATASET_CONFIG_MAP[dataset]
    config = load_yaml_config(config_path)

    suffix = output_suffix or dataset
    resolved_output_dir = output_dir or f"./outputs/umichslm-sft-{suffix}"
    resolved_logging_dir = logging_dir or f"./logs/umichslm-sft-{suffix}"

    config["dataset"] = dataset
    config["output_dir"] = resolved_output_dir
    config["logging_dir"] = resolved_logging_dir
    config["report_to"] = "wandb"

    overrides = {
        "max_samples": max_samples,
        "model_name_or_path": model_name_or_path,
        "eval_dataset": eval_dataset,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "save_steps": save_steps,
        "eval_steps": eval_steps,
        "logging_steps": logging_steps,
        "run_name": run_name or f"umichslm-sft-{suffix}",
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    return config


def run_sft(
    dataset: str,
    base_config: str | None = None,
    max_samples: int | None = None,
    output_suffix: str | None = None,
    output_dir: str | None = None,
    logging_dir: str | None = None,
    generated_config_dir: str = "generated-slurm/sft-configs",
    dry_run: bool = False,
    keep_config: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    **overrides: Any,
) -> Path:
    """Materialize a LLaMA Factory config and optionally launch training."""
    config = resolve_sft_config(
        dataset=dataset,
        base_config=base_config,
        max_samples=max_samples,
        output_suffix=output_suffix,
        output_dir=output_dir,
        logging_dir=logging_dir,
        **overrides,
    )

    suffix = output_suffix or dataset
    config_dir = Path(generated_config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = config_dir / f"sft_{suffix}.yaml"
    dump_yaml(config, temp_config_path)

    cmd = ["python", "scripts/llamafactory_train.py", "train", str(temp_config_path)]
    if dry_run:
        print(f"Resolved config: {temp_config_path}")
        if wandb_project:
            print(f"WANDB_PROJECT={wandb_project}")
        if wandb_entity:
            print(f"WANDB_ENTITY={wandb_entity}")
        print(shell_join(cmd))
        return temp_config_path

    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["logging_dir"]).mkdir(parents=True, exist_ok=True)
    print(f"Created output dir:  {config['output_dir']}")
    print(f"Created logging dir: {config['logging_dir']}")
    print(f"Launching SFT training with dataset: {dataset}")

    env = os.environ.copy()
    if wandb_project:
        env["WANDB_PROJECT"] = wandb_project
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    subprocess.run(cmd, check=True, env=env)

    if not keep_config:
        temp_config_path.unlink(missing_ok=True)
    return temp_config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT training")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--base_config", default=None, help="Base YAML config to use")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_suffix", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--logging_dir", default=None)
    parser.add_argument("--generated_config_dir", default="generated-slurm/sft-configs")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--eval_dataset", default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--keep_config", action="store_true")

    args = parser.parse_args()
    run_sft(**vars(args))


if __name__ == "__main__":
    main()
