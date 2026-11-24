"""Tests for config-driven SLURM workflow rendering."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_pretrain_slurm import render_pretrain_script
from generate_sft_slurm import render_sft_script
from slurm_common import render_environment, render_sbatch_header


def test_sbatch_header_renders_cluster_resources():
    header = render_sbatch_header(
        {
            "partition": "gpu-rtx6000",
            "time": "1:00:00",
            "account": "<account>",
            "nodes": 1,
            "ntasks_per_node": 1,
            "cpus_per_task": 4,
            "mem": "64G",
            "gpus": 1,
        },
        "unit-job",
        "/home/%u/umich-slm/slurm-logs/%x-%j.out",
    )

    assert "#SBATCH --job-name=unit-job" in header
    assert "#SBATCH --partition=gpu-rtx6000" in header
    assert "#SBATCH --account=<account> in header
    assert "#SBATCH --gpus=1" in header
    assert "#SBATCH --mem=64G" in header
    assert "/home/%u/umich-slm/slurm-logs/%x-%j.out" in header


def test_environment_renders_conda_and_exports():
    rendered = render_environment(
        {
            "project_dir": "$HOME/umich-slm",
            "slurm": {"conda_env_name": "umich-slm", "log_dir": "/home/%u/logs"},
            "env": {"HF_HOME": "/path/to/data", "TOKENIZERS_PARALLELISM": "false"},
            "required_env": ["WANDB_API_KEY"],
        }
    )

    assert 'source "$(conda info --base)/etc/profile.d/conda.sh"' in rendered
    assert "conda activate umich-slm" in rendered
    assert 'export PROJECT_DIR="$HOME/umich-slm"' in rendered
    assert "mkdir -p slurm-logs" in rendered
    assert "export HF_HOME=/path/to/data" in rendered
    assert "export TOKENIZERS_PARALLELISM=false" in rendered
    assert ': "${WANDB_API_KEY:?Set WANDB_API_KEY before submitting this job}"' in rendered
    assert "export WANDB_API_KEY" in rendered


def test_environment_renders_conda_env_path():
    rendered = render_environment(
        {
            "slurm": {
                "conda_env_path": "/path/to/data envs/umich-slm",
                "log_dir": "slurm-logs",
            }
        }
    )

    assert "conda activate '/path/to/data envs/umich-slm'" in rendered


def test_pretrain_script_contains_training_command():
    script = render_pretrain_script(
        {
            "project_dir": "$HOME/umich-slm",
            "slurm": {"job_name": "pretrain-test", "log_dir": "slurm-logs", "gpus": 1},
            "data": {
                "dataset": "train_data",
                "dataset_dir": "/path/to/data",
                "eval_dataset": "val_data",
                "do_eval": True,
            },
            "outputs": {"root": "/path/to/data", "run_prefix": "toy"},
            "wandb": {"project": "umich-slm", "entity": "umich-slm"},
            "training": {
                "size": "qwen3_like_tiny",
                "target_tokens_b": 0.02,
                "batch_size": 4,
                "gradient_accumulation_steps": 8,
                "learning_rate": 0.0003,
                "save_steps": 100,
                "eval_steps": 100,
                "logging_steps": 10,
            },
        }
    )

    assert "python scripts/pretrain_fineweb.py" in script
    assert "--dataset train_data" in script
    assert "--eval_dataset val_data" in script
    assert "--target_tokens_b 0.02" in script
    assert "--output_dir /path/to/data" in script
    assert "--wandb_project umich-slm" in script
    assert "--wandb_entity umich-slm" in script


def test_sft_array_maps_learning_rates():
    script = render_sft_script(
        {
            "project_dir": "$HOME/umich-slm",
            "generated_config_dir": "generated-slurm/sft-configs",
            "slurm": {"job_name": "sft-test", "log_dir": "slurm-logs", "gpus": 1},
            "outputs": {"root": "/path/to/data", "run_prefix": "sft"},
            "wandb": {"project": "sft-project", "entity": "umich-slm"},
            "sft": {
                "dataset": "umichslm_sft_train",
                "eval_dataset": "umichslm_sft_val",
                "base_config": "configs/llama_factory/umichslm_sft.yaml",
                "model_name_or_path": "/path/to/data",
                "learning_rate": 2e-5,
            },
            "sweep": {"learning_rate": [1e-5, 2e-5, 5e-5]},
        }
    )

    assert "#SBATCH --array=0-2" in script
    assert "LEARNING_RATE=1e-05" in script
    assert "LEARNING_RATE=2e-05" in script
    assert "LEARNING_RATE=5e-05" in script
    assert "python scripts/run_sft.py" in script
    assert "--dataset umichslm_sft_train" in script
    assert "--eval_dataset umichslm_sft_val" in script
