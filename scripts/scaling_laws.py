#!/usr/bin/env python3
"""
Scaling Laws Experiment Generator for UMichSLM.

This script helps study scaling laws similar to the Chinchilla paper (Hoffmann et al., 2022)
by generating model configurations of varying sizes and training data budgets.

Key concepts from Chinchilla:
- Compute budget C ≈ 6ND (forward + backward pass for transformers)
- Optimal scaling: N* ∝ C^0.5, D* ∝ C^0.5 (roughly equal scaling)
- Loss scales as L(N,D) ≈ E + A/N^α + B/D^β

This script:
1. Generates a grid of model sizes (varying layers, width)
2. Computes parameter counts and estimated FLOPs
3. Creates training configurations for different data sizes
4. Submits SLURM jobs for parallel training

Model presets follow docs/architecture_presets.md (Qwen3 / Qwen3.5, OpenELM LWS,
Hymba, Baguettotron, MobileLLM + LS, MoD, Griffin); see ARCHITECTURE_PRESET_NAMES in this file.

Usage:
    # List available model configs
    python scripts/scaling_laws.py list_models

    # Generate Chinchilla-style experiments (fix C, derive D for each model)
    python scripts/scaling_laws.py chinchilla --output_dir ./chinchilla_experiments

    # Chinchilla with custom models and budgets
    python scripts/scaling_laws.py chinchilla --output_dir ./chinchilla_experiments \
        --models mod_tiny,mod_xsmall,mod_small --compute_budgets 1e15,1e16,1e17

    # Generate a full grid of scaling experiments
    python scripts/scaling_laws.py generate --output_dir ./scaling_experiments

    # Submit all experiments as SLURM jobs
    python scripts/scaling_laws.py submit --output_dir ./scaling_experiments --slurm_config configs/slurm/cluster.yaml

    # Check status of submitted jobs
    python scripts/scaling_laws.py status --output_dir ./scaling_experiments

    # Show Chinchilla-optimal model/data pairs
    python scripts/scaling_laws.py optimal --compute_budgets 1e14,1e15,1e16
"""

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import itertools
import importlib.util

_PRESETS_PATH = Path(__file__).resolve().parent.parent / "src" / "umich_slm" / "configs" / "presets.py"
_spec = importlib.util.spec_from_file_location("umich_slm_presets", _PRESETS_PATH)
_presets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_presets)

get_preset_info = _presets.get_preset_info
estimate_parameters = _presets.estimate_parameters
list_available_sizes = _presets.list_available_sizes



# ============================================================================
# SLURM Configuration
# ============================================================================

@dataclass
class SlurmConfig:
    """SLURM cluster configuration."""
    # Basic SLURM settings
    partition: str = "gpu"
    time: str = "24:00:00"
    gpus: Optional[str] = None
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: int = 8
    mem: Optional[str] = None  # e.g., "64G"

    # Output settings — override via slurm_config JSON
    output_dir: str = "./slurm-logs"
    output_pattern: str = "scaling-%j.out"

    # Account/QOS (cluster-specific)
    account: Optional[str] = None
    qos: Optional[str] = None

    # Email notifications (uniqname for @umich.edu; set to null in JSON to disable)
    mail_account: Optional[str] = None
    mail_type: Optional[str] = "END,FAIL"

    # Environment setup
    conda_env: Optional[str] = None
    modules: List[str] = field(default_factory=list)

    # Environment variables
    env_vars: Dict[str, str] = field(default_factory=dict)

    # Host-specific environment setup (hostname -> env commands)
    host_env: Dict[str, List[str]] = field(default_factory=dict)

    # Data paths — override via slurm_config JSON
    dataset: str = ""
    dataset_dir: str = ""
    eval_dataset: str = ""
    dataset_configs: Dict[str, Dict[str, str]] = field(default_factory=dict)
    models_output_dir: str = "./results"

    @classmethod
    def _filter_fields(cls, data: dict) -> dict:
        valid = {f.name for f in fields(cls)}
        return {k: v for k, v in data.items() if k in valid}

    @classmethod
    def from_yaml(cls, path: str) -> "SlurmConfig":
        """Load SLURM config from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**cls._filter_fields(data))

    @classmethod
    def from_json(cls, path: str) -> "SlurmConfig":
        """Load SLURM config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**cls._filter_fields(data))

    def to_yaml(self, path: str) -> None:
        """Save SLURM config to YAML file."""
        import yaml
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_json(self, path: str) -> None:
        """Save SLURM config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a model size variant."""
    name: str
    config_dict: Dict[str, Any]
    total_params: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    num_key_value_heads: Optional[int] = None  # None = MHA
    # Repeating layer pattern for hybrid models e.g. ["delta", "delta", "delta", "attn"]
    # None means all-attention (standard transformer)
    hybrid_attention_ratio: Optional[List[str]] = None

    @property
    def num_params(self) -> int:
        """Estimate total backbone parameters (non-embedding).

        For hybrid models (e.g. qwen3_5_like) the architecture alternates between
        gated DeltaNet layers and standard attention layers. DeltaNet layers have
        an extra beta gate scalar per head compared to standard attention.
        """
        h = self.hidden_size
        L = self.num_hidden_layers
        i = self.intermediate_size
        n_heads = self.num_attention_heads
        n_kv = self.num_key_value_heads or n_heads
        head_dim = h // n_heads

        # Q, K, V projections + output projection
        attn_params = h * h + 2 * h * (n_kv * head_dim) + h * h

        # SwiGLU MLP: gate, up, down projections
        mlp_params = 3 * h * i

        # RMSNorm: pre-attn + pre-MLP per layer
        norm_params = 2 * h

        if self.hybrid_attention_ratio is not None:
            pattern = self.hybrid_attention_ratio
            n_attn = sum(1 for t in pattern if t == "attn")
            full_repeats, remainder = divmod(L, len(pattern))
            total_attn_layers = full_repeats * n_attn + sum(
                1 for t in pattern[:remainder] if t == "attn"
            )
            total_delta_layers = L - total_attn_layers
            # DeltaNet layers carry an extra beta gate scalar per head
            delta_extra = n_heads
        else:
            total_attn_layers = L
            total_delta_layers = 0
            delta_extra = 0

        attn_layer_params = attn_params + mlp_params + norm_params
        delta_layer_params = attn_params + delta_extra + mlp_params + norm_params
        total = (
            total_attn_layers * attn_layer_params
            + total_delta_layers * delta_layer_params
            + h  # final RMSNorm
        )
        return total

    @property
    def num_params_with_embeddings(self) -> int:
        """Estimate total parameters including embeddings."""
        vocab_size = 49152  # SmolLM2 tokenizer
        embedding_params = vocab_size * self.hidden_size
        return self.num_params + embedding_params

    def flops_per_token(self) -> int:
        """Estimate FLOPs per token for training (forward + backward)."""
        return 6 * self.num_params

    def estimate_training_flops(self, num_tokens: int) -> float:
        """Estimate total training FLOPs for a given number of tokens."""
        return self.flops_per_token() * num_tokens


@dataclass
class ScalingExperiment:
    """A single scaling law experiment configuration."""
    model_config: ModelConfig
    num_tokens: int  # Training tokens
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    dataset: str
    dataset_dir: str
    eval_dataset: str
    partition: Optional[str] = None # per-experiment partition override
    job_id: Optional[str] = None    # SLURM job ID after submission

    @property
    def experiment_name(self) -> str:
        params_m = self.model_config.num_params / 1e6
        tokens_m = self.num_tokens / 1e6
        return f"{self.model_config.name}_{params_m:.0f}M_{tokens_m:.0f}Mtok"

    @property
    def estimated_flops(self) -> float:
        return self.model_config.estimate_training_flops(self.num_tokens)


# ============================================================================
# Predefined Model Configurations
# ============================================================================

# Names aligned with docs/architecture_presets.md (all documented preset families).
ARCHITECTURE_PRESET_NAMES: tuple[str, ...] = (
    # Qwen3-like
    "qwen3_like_tiny",
    "qwen3_like_xsmall",
    "qwen3_like_small",
    # Qwen3.5-like (DeltaNet hybrid)
    "qwen3_5_like_tiny",
    "qwen3_5_like_xsmall",
    "qwen3_5_like_small",
    # Open-ELM layer-wise scaling
    "openelm_like_tiny",
    "openelm_like_xsmall",
    "openelm_like_small",
    # Hymba-like (KV sharing)
    "hymba_like_tiny",
    "hymba_like_xsmall",
    "hymba_like_small",
    # Baguettotron-like
    "baguette_micro",
    "baguette_tiny",
    "baguette_small",
    # MobileLLM (+ LS weight sharing)
    "mobilellm_tiny",
    "mobilellm_tiny_ls",
    "mobilellm_xsmall",
    "mobilellm_xsmall_ls",
    "mobilellm_small",
    "mobilellm_small_ls",
    # Mixture-of-Depths
    "mod_tiny",
    "mod_xsmall",
    "mod_small",
    # Griffin hybrid
    "Griffin_tiny",
    "Griffin_xsmall",
    "Griffin_small",
)


def _build_architecture_model_configs() -> Dict[str, ModelConfig]:
    available = set(list_available_sizes())
    missing = [n for n in ARCHITECTURE_PRESET_NAMES if n not in available]
    if missing:
        raise RuntimeError(
            "scaling_laws.py ARCHITECTURE_PRESET_NAMES includes presets not in presets.py: "
            + ", ".join(missing)
        )
    return {
        name: ModelConfig(
            name=name,
            config_dict=get_preset_info(name),
            total_params=estimate_parameters(name),
        )
        for name in ARCHITECTURE_PRESET_NAMES
    }


MODEL_CONFIGS = _build_architecture_model_configs()


def _format_heads_for_list(cfg: Dict[str, Any]) -> str:
    if cfg.get("layer_wise_scaling") and cfg.get("head_scaling_range"):
        lo, hi = cfg["head_scaling_range"]
        return f"{lo}-{hi}"
    h = cfg.get("num_attention_heads")
    return str(h) if h is not None else "-"


def _format_mlp_for_list(cfg: Dict[str, Any]) -> str:
    if cfg.get("layer_wise_scaling") and cfg.get("ffn_multiplier_range"):
        lo, hi = cfg["ffn_multiplier_range"]
        return f"ffn×{lo}-{hi}"
    m = cfg.get("intermediate_size")
    return str(m) if m is not None else "-"

DATASET_CONFIGS = {
    20: {
        "dataset": "mix_code10_math5_math_reasoning5_textbook20_web60_train_20M",
        "dataset_dir": "/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data/toy_datasets/20M",
        "eval_dataset": "mix_code10_math5_math_reasoning5_textbook20_web60_val_0.6M",
    },
    40: {
        "dataset": "mix_code10_math5_math_reasoning5_textbook20_web60_train_40M",
        "dataset_dir": "/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data/toy_datasets/40M",
        "eval_dataset": "mix_code10_math5_math_reasoning5_textbook20_web60_val_1.2M",
    },
    200: {
        "dataset": "mix_code10_math5_math_reasoning5_textbook20_web60_train_200M",
        "dataset_dir": "/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data/toy_datasets/200M",
        "eval_dataset": "mix_code10_math5_math_reasoning5_textbook20_web60_val_6M",
    },
    500: {
        "dataset": "mix_code10_math5_math_reasoning5_textbook20_web60_train_500M",
        "dataset_dir": "/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data/toy_datasets/500M",
        "eval_dataset": "mix_code10_math5_math_reasoning5_textbook20_web60_val_15M",
    },
    1000: {
        "dataset": "mix_code10_math5_math_reasoning5_textbook20_web60_train_1000M",
        "dataset_dir": "/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data/toy_datasets/1B",
        "eval_dataset": "mix_code10_math5_math_reasoning5_textbook20_web60_val_30M",
    },
}

TOKEN_COUNTS_M = [20, 40, 200, 500, 1000]  # Training tokens in millions


# ============================================================================
# Hyperparameter Computation
# ============================================================================

def compute_optimal_lr(num_params: int, base_lr: float = 3e-4) -> float:
    """Compute learning rate scaled by model size."""
    base_params = 125e6
    scale = math.sqrt(base_params / num_params)
    return max(1e-5, min(1e-3, base_lr * scale))


def compute_batch_size(num_params: int, max_batch: int = 32) -> tuple[int, int]:
    """Compute batch size and gradient accumulation for model size.

    Large vocab (e.g. 49k) makes LM-head / CE memory scale with batch×seq×vocab; keep
    per-device batch moderate for tens-of-millions-parameter models even when the
    backbone is small.
    """
    if num_params < 20e6:
        batch_size = min(64, max_batch)
    elif num_params < 50e6:
        batch_size = min(16, max_batch)
    elif num_params < 100e6:
        batch_size = min(24, max_batch)
    else:
        batch_size = min(16, max_batch)

    target_effective = 128
    grad_accum = max(1, target_effective // batch_size)

    return batch_size, grad_accum


# ============================================================================
# Experiment Generation
# ============================================================================

def generate_scaling_grid(
    model_names: Optional[List[str]] = None,
    token_counts_m: Optional[List[int]] = None,
    compute_budget: Optional[float] = None,
    slurm_config: Optional[SlurmConfig] = None,
) -> List[ScalingExperiment]:
    """Generate a grid of scaling experiments."""
    if model_names is None:
        model_names = list(MODEL_CONFIGS.keys())

    if token_counts_m is None:
        token_counts_m = TOKEN_COUNTS_M

    experiments = []

    for model_name in model_names:
        if model_name not in MODEL_CONFIGS:
            print(f"Warning: Unknown model config '{model_name}', skipping")
            continue

        model_config = MODEL_CONFIGS[model_name]

        for tokens_m in token_counts_m:
            num_tokens = tokens_m * 1_000_000

            estimated_flops = model_config.estimate_training_flops(num_tokens)
            if compute_budget and estimated_flops > compute_budget:
                continue

            lr = compute_optimal_lr(model_config.num_params)
            batch_size, grad_accum = compute_batch_size(model_config.num_params)
            dataset_cfg = DATASET_CONFIGS.get(tokens_m, {})
            if slurm_config is not None:
                dataset_cfg = {
                    **dataset_cfg,
                    **slurm_config.dataset_configs.get(str(tokens_m), {}),
                }

            experiment = ScalingExperiment(
                model_config=model_config,
                num_tokens=num_tokens,
                batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=lr,
                dataset=dataset_cfg.get("dataset", slurm_config.dataset if slurm_config else ""),
                dataset_dir=dataset_cfg.get(
                    "dataset_dir", slurm_config.dataset_dir if slurm_config else ""
                ),
                eval_dataset=dataset_cfg.get(
                    "eval_dataset", slurm_config.eval_dataset if slurm_config else ""
                ),
            )
            experiments.append(experiment)

    return experiments


def generate_chinchilla_optimal_pairs(
    compute_budgets: List[float],
) -> List[Dict[str, Any]]:
    """Generate Chinchilla-optimal (N, D) pairs for given compute budgets."""
    results = []

    for C in compute_budgets:
        N_optimal = math.sqrt(C / 6)
        D_optimal = C / (6 * N_optimal)

        closest_model = None
        closest_diff = float('inf')
        for model in MODEL_CONFIGS.values():
            diff = abs(model.num_params - N_optimal)
            if diff < closest_diff:
                closest_diff = diff
                closest_model = model

        results.append({
            "compute_budget_flops": C,
            "compute_budget_pflops": C / 1e15,
            "optimal_params": int(N_optimal),
            "optimal_tokens": int(D_optimal),
            "suggested_model": closest_model.name if closest_model else None,
            "suggested_model_params": closest_model.num_params if closest_model else None,
        })

    return results


# ============================================================================
# Chinchilla-Style Experiment Generation
# ============================================================================

# Reasonable default budgets (FLOPs) that span tiny, xsmall, and small architecture preset sizes.
# Picked so multiple model sizes overlap at the same budget for iso-FLOP comparison.
# N = total_params (including embeddings), C = 6 * N * D.
#
#   Budget      | mod_tiny (30M) | mod_xsmall (78M) | mod_small (252M)
#   ------------|----------------|------------------|------------------
#   1e15        |  ~6M tok       |  —               |  —
#   5e15        |  ~28M tok      |  —               |  —
#   1e16        |  ~55M tok      |  ~21M tok        |  —
#   5e16        |  ~276M tok     | ~107M tok        |  ~33M tok
#   1e17        |  ~551M tok     | ~213M tok        |  ~66M tok
#   5e17        |  —             |  —               | ~331M tok
#   1e18        |  —             |  —               | ~662M tok
DEFAULT_CHINCHILLA_BUDGETS = [1e15, 5e15, 1e16, 5e16, 1e17, 5e17, 1e18]

DEFAULT_CHINCHILLA_MODELS = ["mod_tiny", "mod_xsmall", "mod_small"]

# tiny/xsmall/micro run on L40, small runs on RTX 6000
GPU_BY_SIZE = {
    "micro": {"partition": "spgpu"},
    "tiny": {"partition": "spgpu"},
    "xsmall": {"partition": "spgpu"},
    "small": {"partition": "gpu-rtx6000"},
}


def _chinchilla_size_suffix(model_name: str) -> str:
    """Map preset name to coarse size for partition hints (handles mobilellm_*_ls)."""
    base = model_name[:-3] if model_name.endswith("_ls") else model_name
    return base.rsplit("_", 1)[-1]


def generate_chinchilla_experiments(
    model_names: List[str],
    compute_budgets: List[float],
    slurm_config: Optional[SlurmConfig] = None,
) -> List[ScalingExperiment]:
    """
    Chinchilla-style: fix compute budget C, derive token count D = C / (6*N)
    for each model, then pick the smallest dataset that covers D tokens.

    Trains on exactly D tokens (a subset of the dataset), so parameter sizes
    stay fixed and the token count absorbs the slack.
    """
    available_token_counts = sorted(DATASET_CONFIGS.keys())  # e.g. [20, 40, 200, 500, 1000]
    min_tokens = available_token_counts[0] * 1_000_000
    max_tokens = available_token_counts[-1] * 1_000_000

    experiments = []

    print("\n" + "=" * 100)
    print("Chinchilla Experiment Planning  (C = 6 * N * D)")
    print("=" * 100)
    header = (
        f"{'Budget (FLOPs)':<18} {'Model':<18} {'N (total)':<14} "
        f"{'D (tokens)':<14} {'Dataset':<10} {'Utilization':<12}"
    )
    print(header)
    print("-" * 100)

    for C in sorted(compute_budgets):
        for model_name in model_names:
            if model_name not in MODEL_CONFIGS:
                print(f"  !! unknown model '{model_name}', skipping")
                continue
            
            # calculate token budget
            model_config = MODEL_CONFIGS[model_name]
            N = model_config.total_params
            D = C / (6 * N)

            if D < min_tokens * 0.5:
                print(f"{C:<18.1e} {model_name:<18} {N/1e6:<14.1f}M {D/1e6:<14.1f}M {'—':<10} skip: D too small")
                continue
            if D > max_tokens * 1.1:
                print(f"{C:<18.1e} {model_name:<18} {N/1e6:<14.1f}M {D/1e6:<14.1f}M {'—':<10} skip: D too large")
                continue

            D = min(D, float(max_tokens))

            # Pick smallest dataset that has >= D tokens
            dataset_tokens_m = available_token_counts[-1]
            for t_m in available_token_counts:
                if t_m * 1_000_000 >= D:
                    dataset_tokens_m = t_m
                    break

            utilization = D / (dataset_tokens_m * 1_000_000)

            dataset_cfg = dict(DATASET_CONFIGS[dataset_tokens_m])
            if slurm_config is not None:
                dataset_cfg.update(slurm_config.dataset_configs.get(str(dataset_tokens_m), {}))

            lr = compute_optimal_lr(N)
            batch_size, grad_accum = compute_batch_size(N)

            # Look up GPU/partition by size suffix (e.g. "mod_tiny" → "tiny", "mobilellm_tiny_ls" → "tiny")
            size_suffix = _chinchilla_size_suffix(model_name)
            gpu_cfg = GPU_BY_SIZE.get(size_suffix, {})

            exp = ScalingExperiment(
                model_config=model_config,
                num_tokens=int(D),
                batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=lr,
                dataset=dataset_cfg.get("dataset", slurm_config.dataset if slurm_config else ""),
                dataset_dir=dataset_cfg.get("dataset_dir", slurm_config.dataset_dir if slurm_config else ""),
                eval_dataset=dataset_cfg.get("eval_dataset", slurm_config.eval_dataset if slurm_config else ""),
                partition=gpu_cfg.get("partition"),
            )
            experiments.append(exp)

            print(
                f"{C:<18.1e} {model_name:<18} {N/1e6:<14.1f}M "
                f"{D/1e6:<14.1f}M {dataset_tokens_m:<10}M {utilization:<12.1%}"
            )

    print("=" * 100)
    print(f"\nTotal Chinchilla experiments: {len(experiments)}")
    return experiments


# ============================================================================
# Script Generation and SLURM Submission
# ============================================================================

def generate_slurm_script(
    exp: ScalingExperiment,
    slurm_config: SlurmConfig,
    script_dir: Path,
) -> Path:
    """Generate a SLURM script for an experiment."""
    config = exp.model_config
    tokens_b = exp.num_tokens / 1e9

    # Build SLURM header
    partition = exp.partition or slurm_config.partition
    log_dir = slurm_config.output_dir
    log_file = f"{log_dir}/{exp.experiment_name}-%j.out"
    err_file = f"{log_dir}/{exp.experiment_name}-%j.err"
    slurm_header = f"""#!/bin/bash
#SBATCH --job-name=scale-{exp.experiment_name}
#SBATCH --output={log_file}
#SBATCH --error={err_file}
#SBATCH --partition={partition}
#SBATCH --time={slurm_config.time}
#SBATCH --nodes={slurm_config.nodes}
#SBATCH --ntasks-per-node={slurm_config.ntasks_per_node}
#SBATCH --cpus-per-task={slurm_config.cpus_per_task}"""

    if slurm_config.gpus:
        slurm_header += f"\n#SBATCH --gpus={slurm_config.gpus}"

    if slurm_config.mem:
        slurm_header += f"\n#SBATCH --mem={slurm_config.mem}"
    if slurm_config.account:
        slurm_header += f"\n#SBATCH --account={slurm_config.account}"
    if slurm_config.qos:
        slurm_header += f"\n#SBATCH --qos={slurm_config.qos}"
    if slurm_config.mail_account:
        slurm_header += f"\n#SBATCH --mail-user={slurm_config.mail_account}@umich.edu"
        if slurm_config.mail_type:
            slurm_header += f"\n#SBATCH --mail-type={slurm_config.mail_type}"

    # Build environment setup
    env_setup = f"""
# Scaling Law Experiment: {exp.experiment_name}
# Model: {config.name} ({config.num_params/1e6:.1f}M params)
# Training tokens: {exp.num_tokens/1e6:.0f}M
# Estimated FLOPs: {exp.estimated_flops:.2e}

echo "========================================"
echo "Experiment: {exp.experiment_name}"
echo "Model:      {config.name} ({config.num_params/1e6:.1f}M params)"
echo "Tokens:     {exp.num_tokens/1e6:.0f}M"
echo "FLOPs:      {exp.estimated_flops:.2e}"
echo "Host:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:      $(date)"
echo "Job ID:     $SLURM_JOB_ID"
echo "========================================"

"""

    # Module loads
    if slurm_config.modules:
        env_setup += "# Load modules\n"
        for module in slurm_config.modules:
            env_setup += f"module load {module}\n"
        env_setup += "\n"

    # Conda environment (always activate umich-slm by default)
    conda_env_name = slurm_config.conda_env or "umich-slm"
    env_setup += (
        "# Initialize and activate conda environment\n"
        'source "$(conda info --base)/etc/profile.d/conda.sh"\n'
        f"conda activate {conda_env_name}\n\n"
    )

    # Project directory
    env_setup += (
        "# Project directory\n"
        'export PROJECT_DIR="$HOME/umich-slm"\n'
        'cd "$PROJECT_DIR"\n\n'
    )

    # Host-specific environment
    if slurm_config.host_env:
        env_setup += "# Host-specific environment setup\n"
        for hostname, commands in slurm_config.host_env.items():
            env_setup += f'if [[ $(hostname) == "{hostname}"* ]]; then\n'
            for cmd in commands:
                env_setup += f"  {cmd}\n"
            env_setup += "fi\n\n"

    # Environment variables
    env_vars_str = " \\\n  ".join(f"{k}={v}" for k, v in slurm_config.env_vars.items())

    # Training command
    # --val_percent 0: all D tokens go to training; evaluation uses --eval_dataset separately
    training_cmd = f"""{env_vars_str} \\
  python scripts/pretrain_fineweb.py \\
  --output_dir {slurm_config.models_output_dir}/{exp.experiment_name}/ \\
  --size {config.name} \\
  --batch_size {exp.batch_size} \\
  --gradient_accumulation_steps {exp.gradient_accumulation_steps} \\
  --learning_rate {exp.learning_rate:.2e} \\
  --target_tokens_b {tokens_b:.3f} \\
  --val_percent 0 \\
  --dataset {exp.dataset} \\
  --dataset_dir {exp.dataset_dir} \\
  --do_eval --eval_dataset {exp.eval_dataset} --eval_steps 200
"""

    footer = f"""
EXIT_CODE=$?
echo "========================================"
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "========================================"
exit $EXIT_CODE
"""

    script_content = slurm_header + env_setup + training_cmd + footer

    # Save script
    script_path = script_dir / f"{exp.experiment_name}.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    return script_path


def submit_slurm_job(script_path: Path, dry_run: bool = False) -> Optional[str]:
    """Submit a SLURM job and return the job ID."""
    if dry_run:
        print(f"  [DRY RUN] Would submit: sbatch {script_path}")
        return "DRY_RUN"

    try:
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse job ID from output like "Submitted batch job 12345"
        output = result.stdout.strip()
        if "Submitted batch job" in output:
            job_id = output.split()[-1]
            return job_id
        return None
    except subprocess.CalledProcessError as e:
        print(f"  Error submitting job: {e.stderr}")
        return None
    except FileNotFoundError:
        print("  Error: sbatch command not found. Is SLURM installed?")
        return None


def check_job_status(job_ids: List[str]) -> Dict[str, str]:
    """Check status of SLURM jobs."""
    if not job_ids:
        return {}

    try:
        result = subprocess.run(
            ["squeue", "-j", ",".join(job_ids), "-h", "-o", "%i %t"],
            capture_output=True,
            text=True,
        )
        status = {}
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    status[parts[0]] = parts[1]

        # Mark jobs not in queue as completed/unknown
        for job_id in job_ids:
            if job_id not in status:
                # Check if job completed
                sacct_result = subprocess.run(
                    ["sacct", "-j", job_id, "-n", "-o", "State", "-P"],
                    capture_output=True,
                    text=True,
                )
                if sacct_result.stdout.strip():
                    state = sacct_result.stdout.strip().split("\n")[0]
                    status[job_id] = state
                else:
                    status[job_id] = "UNKNOWN"

        return status
    except FileNotFoundError:
        print("Error: squeue/sacct commands not found. Is SLURM installed?")
        return {job_id: "UNKNOWN" for job_id in job_ids}


# ============================================================================
# Export and Reporting Functions
# ============================================================================

def export_experiment_configs(
    experiments: List[ScalingExperiment],
    output_dir: Path,
    slurm_config: SlurmConfig,
) -> List[Dict[str, Any]]:
    """Export experiment configurations as shell scripts."""
    output_dir = Path(output_dir)
    scripts_dir = output_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for exp in experiments:
        config = exp.model_config
        seq_len = 2048
        effective_batch = exp.batch_size * exp.gradient_accumulation_steps
        tokens_per_step = effective_batch * seq_len
        max_steps = exp.num_tokens // tokens_per_step

        script_path = generate_slurm_script(exp, slurm_config, scripts_dir)

        summary.append({
            "name": exp.experiment_name,
            "model": config.name,
            "params_m": config.num_params / 1e6,
            "tokens_m": exp.num_tokens / 1e6,
            "dataset": exp.dataset,
            "dataset_dir": exp.dataset_dir,
            "eval_dataset": exp.eval_dataset,
            "flops": exp.estimated_flops,
            "pflops": exp.estimated_flops / 1e15,
            "batch_size": exp.batch_size,
            "grad_accum": exp.gradient_accumulation_steps,
            "lr": exp.learning_rate,
            "max_steps": max_steps,
            "script": str(script_path),
            "job_id": None,
            "status": "pending",
        })

    # Save summary JSON
    summary_path = output_dir / "experiments_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save SLURM config for reference
    slurm_config.to_json(output_dir / "slurm_config.json")

    print(f"Generated {len(experiments)} experiment configs in {output_dir}")
    print(f"Scripts saved to {scripts_dir}")
    print(f"Summary saved to {summary_path}")

    return summary


def list_model_configs():
    """Print all available model configurations."""
    print("\n" + "=" * 90)
    print("Available Model Configurations for Scaling Experiments (docs/architecture_presets.md)")
    print("=" * 90)
    print(
        f"{'Name':<24} {'Hidden':<8} {'Layers':<8} {'Heads':<10} "
        f"{'MLP / FFN':<12} {'Total Params':<14}"
    )
    print("-" * 90)

    for name, config in sorted(MODEL_CONFIGS.items(), key=lambda x: x[1].num_params):
        cfg = config.config_dict
        params_m = config.num_params / 1e6
        heads_s = _format_heads_for_list(cfg)
        mlp_s = _format_mlp_for_list(cfg)
        params_str = f"{params_m:.1f}M"
        print(
            f"{name:<24} "
            f"{cfg['hidden_size']:<8} "
            f"{cfg['num_hidden_layers']:<8} "
            f"{heads_s:<10} "
            f"{mlp_s:<12} "
            f"{params_str:<14}"
        )

    print("\n" + "=" * 90)
    print("Token Counts for Scaling (millions):", TOKEN_COUNTS_M)
    print("=" * 90 + "\n")



def print_scaling_summary(experiments: List[ScalingExperiment]):
    """Print a summary table of scaling experiments."""
    print("\n" + "=" * 100)
    print("Scaling Law Experiments Summary")
    print("=" * 100)
    print(f"{'Experiment':<35} {'Params':<12} {'Tokens':<12} {'FLOPs':<15} {'LR':<10} {'Batch':<8}")
    print("-" * 100)

    total_flops = 0
    for exp in sorted(experiments, key=lambda x: x.estimated_flops):
        params_m = exp.model_config.num_params / 1e6
        tokens_m = exp.num_tokens / 1e6
        flops_str = f"{exp.estimated_flops:.2e}"
        effective_batch = exp.batch_size * exp.gradient_accumulation_steps

        print(f"{exp.experiment_name:<35} {params_m:<12.1f}M {tokens_m:<12.0f}M "
              f"{flops_str:<15} {exp.learning_rate:<10.2e} {effective_batch:<8}")
        total_flops += exp.estimated_flops

    print("-" * 100)
    print(f"Total experiments: {len(experiments)}")
    print(f"Total compute: {total_flops:.2e} FLOPs ({total_flops/1e15:.2f} PFLOPs)")
    print("=" * 100 + "\n")


# ============================================================================
# CLI Argument Parsing
# ============================================================================

def add_slurm_args(parser: argparse.ArgumentParser):
    """Add SLURM-related arguments to a parser."""
    slurm_group = parser.add_argument_group("SLURM Configuration")
    slurm_group.add_argument("--slurm_config", type=str, default=None,
                            help="Path to SLURM config file (YAML or JSON)")
    slurm_group.add_argument("--partition", type=str, default=None,
                            help="SLURM partition (overrides config file)")
    slurm_group.add_argument("--time", type=str, default=None,
                            help="Job time limit (overrides config file)")
    slurm_group.add_argument("--gpus", type=str, default=None,
                            help="GPU count/resource string for --gpus (overrides config file)")
    slurm_group.add_argument("--account", type=str, default=None,
                            help="SLURM account (overrides config file)")
    slurm_group.add_argument("--qos", type=str, default=None,
                            help="SLURM QOS (overrides config file)")
    slurm_group.add_argument("--mem", type=str, default=None,
                            help="Memory per node (overrides config file)")
    slurm_group.add_argument(
        "--mail_account",
        type=str,
        default=None,
        help="Uniqname for Slurm mail (sets --mail-user=<uniqname>@umich.edu); empty string disables",
    )


def load_slurm_config(args) -> SlurmConfig:
    """Load SLURM config from file and apply CLI overrides."""
    if args.slurm_config:
        path = args.slurm_config
        if path.endswith(".yaml") or path.endswith(".yml"):
            config = SlurmConfig.from_yaml(path)
        elif path.endswith(".json"):
            config = SlurmConfig.from_json(path)
        else:
            raise ValueError(f"Unknown config file format: {path}")
    else:
        config = SlurmConfig()

    # Apply CLI overrides
    if args.partition:
        config.partition = args.partition
    if args.time:
        config.time = args.time
    if args.gpus:
        config.gpus = args.gpus
    if args.account:
        config.account = args.account
    if args.qos:
        config.qos = args.qos
    if args.mem:
        config.mem = args.mem
    if getattr(args, "mail_account", None) is not None:
        config.mail_account = args.mail_account or None

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate and submit scaling law experiments for UMichSLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available model configs
    python scripts/scaling_laws.py list_models

    # Generate experiment scripts (no submission)
    python scripts/scaling_laws.py generate --output_dir ./scaling_experiments

    # Generate with custom SLURM config
    python scripts/scaling_laws.py generate --output_dir ./scaling_experiments \\
        --slurm_config configs/slurm/cluster.yaml

    # Submit all experiments as SLURM jobs
    python scripts/scaling_laws.py submit --output_dir ./scaling_experiments \\
        --slurm_config configs/slurm/cluster.yaml

    # Submit with CLI SLURM overrides
    python scripts/scaling_laws.py submit --output_dir ./scaling_experiments \\
        --partition gpu-large --time 72:00:00 --gpus 1

    # Dry run (show what would be submitted)
    python scripts/scaling_laws.py submit --output_dir ./scaling_experiments --dry_run

    # Check status of submitted jobs
    python scripts/scaling_laws.py status --output_dir ./scaling_experiments

    # Chinchilla experiments with SLURM submission
    python scripts/scaling_laws.py chinchilla --output_dir ./chinchilla_experiments \\
        --slurm_config configs/slurm/cluster.yaml --submit

    # Show Chinchilla-optimal model/data pairs
    python scripts/scaling_laws.py optimal --compute_budgets 1e14,1e15,1e16

    # Generate a template SLURM config file
    python scripts/scaling_laws.py init_config --output configs/slurm/my_cluster.yaml
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List models command
    subparsers.add_parser("list_models", help="List available model configurations")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate scaling experiment scripts")
    gen_parser.add_argument("--output_dir", type=str, required=True,
                           help="Output directory for experiment configs")
    gen_parser.add_argument("--models", type=str, default=None,
                           help="Comma-separated list of model names (default: all)")
    gen_parser.add_argument("--tokens", type=str, default=None,
                           help="Comma-separated list of token counts in millions")
    gen_parser.add_argument("--compute_budget", type=float, default=None,
                           help="Maximum compute budget in FLOPs")
    gen_parser.add_argument("--dataset", type=str, default=None,
                           help="Dataset name (overrides config)")
    gen_parser.add_argument("--dataset_dir", type=str, default=None,
                           help="Dataset directory (overrides config)")
    add_slurm_args(gen_parser)

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit scaling experiments as SLURM jobs")
    submit_parser.add_argument("--output_dir", type=str, required=True,
                              help="Output directory with experiment configs")
    submit_parser.add_argument("--models", type=str, default=None,
                              help="Comma-separated list of model names (default: all)")
    submit_parser.add_argument("--tokens", type=str, default=None,
                              help="Comma-separated list of token counts in millions")
    submit_parser.add_argument("--compute_budget", type=float, default=None,
                              help="Maximum compute budget in FLOPs")
    submit_parser.add_argument("--dataset", type=str, default=None,
                              help="Dataset name (overrides config)")
    submit_parser.add_argument("--dataset_dir", type=str, default=None,
                              help="Dataset directory (overrides config)")
    submit_parser.add_argument("--dry_run", action="store_true",
                              help="Show what would be submitted without actually submitting")
    submit_parser.add_argument("--max_jobs", type=int, default=None,
                              help="Maximum number of jobs to submit")
    add_slurm_args(submit_parser)

    # Status command
    status_parser = subparsers.add_parser("status", help="Check status of submitted jobs")
    status_parser.add_argument("--output_dir", type=str, required=True,
                              help="Output directory with experiment configs")

    # Optimal pairs command
    opt_parser = subparsers.add_parser("optimal", help="Show Chinchilla-optimal model/data pairs")
    opt_parser.add_argument("--compute_budgets", type=str, default="1e14,1e15,1e16",
                           help="Comma-separated compute budgets in FLOPs")

    # Chinchilla command
    chin_parser = subparsers.add_parser(
        "chinchilla",
        help="Generate Chinchilla-style scaling experiments (fix C, derive D per model)",
    )
    chin_parser.add_argument("--output_dir", type=str, required=True,
                             help="Output directory for experiment configs")
    chin_parser.add_argument("--models", type=str,
                             default=",".join(DEFAULT_CHINCHILLA_MODELS),
                             help="Comma-separated model names (default: MoD family)")
    chin_parser.add_argument("--compute_budgets", type=str,
                             default=",".join(f"{b:.0e}" for b in DEFAULT_CHINCHILLA_BUDGETS),
                             help="Comma-separated compute budgets in FLOPs")
    chin_parser.add_argument("--submit", action="store_true",
                             help="Also submit SLURM jobs after generating scripts")
    chin_parser.add_argument("--dry_run", action="store_true",
                             help="Show what would be submitted without actually submitting")
    add_slurm_args(chin_parser)

    # Init config command
    init_parser = subparsers.add_parser("init_config", help="Generate a template SLURM config file")
    init_parser.add_argument("--output", type=str, required=True,
                            help="Output path for config file (.yaml or .json)")

    args = parser.parse_args()

    if args.command == "list_models":
        list_model_configs()

    elif args.command == "generate":
        slurm_config = load_slurm_config(args)

        # Apply dataset overrides
        if args.dataset:
            slurm_config.dataset = args.dataset
        if args.dataset_dir:
            slurm_config.dataset_dir = args.dataset_dir

        model_names = [m.strip() for m in args.models.split(",")] if args.models else None
        token_counts = [int(t.strip()) for t in args.tokens.split(",")] if args.tokens else None

        experiments = generate_scaling_grid(
            model_names=model_names,
            token_counts_m=token_counts,
            compute_budget=args.compute_budget,
            slurm_config=slurm_config,
        )

        if not experiments:
            print("No experiments generated. Check your parameters.")
            return

        print_scaling_summary(experiments)
        export_experiment_configs(experiments, Path(args.output_dir), slurm_config)

    elif args.command == "submit":
        slurm_config = load_slurm_config(args)

        # Apply dataset overrides
        if args.dataset:
            slurm_config.dataset = args.dataset
        if args.dataset_dir:
            slurm_config.dataset_dir = args.dataset_dir

        model_names = [m.strip() for m in args.models.split(",")] if args.models else None
        token_counts = [int(t.strip()) for t in args.tokens.split(",")] if args.tokens else None

        experiments = generate_scaling_grid(
            model_names=model_names,
            token_counts_m=token_counts,
            compute_budget=args.compute_budget,
            slurm_config=slurm_config,
        )

        if not experiments:
            print("No experiments to submit. Check your parameters.")
            return

        if args.max_jobs and len(experiments) > args.max_jobs:
            print(f"Limiting to {args.max_jobs} jobs (out of {len(experiments)})")
            experiments = experiments[:args.max_jobs]

        print_scaling_summary(experiments)

        output_dir = Path(args.output_dir)
        summary = export_experiment_configs(experiments, output_dir, slurm_config)

        print("\n" + "=" * 60)
        print("Submitting SLURM Jobs")
        print("=" * 60)

        for i, exp_summary in enumerate(summary):
            script_path = exp_summary["script"]
            print(f"\n[{i+1}/{len(summary)}] {exp_summary['name']}")

            job_id = submit_slurm_job(Path(script_path), dry_run=args.dry_run)
            if job_id:
                exp_summary["job_id"] = job_id
                exp_summary["status"] = "submitted"
                print(f"  Submitted: Job ID {job_id}")
            else:
                exp_summary["status"] = "failed"
                print(f"  Failed to submit")

        # Update summary with job IDs
        summary_path = output_dir / "experiments_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        submitted = sum(1 for s in summary if s["status"] == "submitted")
        print(f"Submitted {submitted}/{len(summary)} jobs")
        print(f"Updated summary: {summary_path}")
        print("=" * 60 + "\n")

    elif args.command == "status":
        output_dir = Path(args.output_dir)
        summary_path = output_dir / "experiments_summary.json"

        if not summary_path.exists():
            print(f"Error: No experiments found at {summary_path}")
            return

        with open(summary_path) as f:
            summary = json.load(f)

        job_ids = [s["job_id"] for s in summary if s.get("job_id") and s["job_id"] != "DRY_RUN"]

        if not job_ids:
            print("No jobs have been submitted yet.")
            return

        print("\n" + "=" * 80)
        print("Scaling Experiments Status")
        print("=" * 80)

        statuses = check_job_status(job_ids)

        status_map = {
            "PD": "PENDING",
            "R": "RUNNING",
            "CG": "COMPLETING",
            "CD": "COMPLETED",
            "F": "FAILED",
            "TO": "TIMEOUT",
            "CA": "CANCELLED",
        }

        print(f"{'Experiment':<40} {'Job ID':<12} {'Status':<15}")
        print("-" * 80)

        for s in summary:
            job_id = s.get("job_id", "N/A")
            if job_id and job_id != "DRY_RUN":
                raw_status = statuses.get(job_id, "UNKNOWN")
                status = status_map.get(raw_status, raw_status)
            elif job_id == "DRY_RUN":
                status = "DRY_RUN"
            else:
                status = "NOT_SUBMITTED"

            print(f"{s['name']:<40} {job_id:<12} {status:<15}")

        print("=" * 80 + "\n")

    elif args.command == "optimal":
        budgets = [float(b.strip()) for b in args.compute_budgets.split(",")]
        pairs = generate_chinchilla_optimal_pairs(budgets)

        print("\n" + "=" * 80)
        print("Chinchilla-Optimal (N*, D*) Pairs")
        print("=" * 80)
        print(f"{'Compute (PFLOPs)':<20} {'Optimal Params':<18} {'Optimal Tokens':<18} {'Suggested Model':<20}")
        print("-" * 80)

        for p in pairs:
            pflops = p['compute_budget_pflops']
            params_m = p['optimal_params'] / 1e6
            tokens_m = p['optimal_tokens'] / 1e6
            model = p['suggested_model'] or "N/A"
            print(f"{pflops:<20.3f} {params_m:<18.1f}M {tokens_m:<18.1f}M {model:<20}")

        print("=" * 80 + "\n")

    elif args.command == "chinchilla":
        slurm_config = load_slurm_config(args)

        model_names = [m.strip() for m in args.models.split(",")]
        compute_budgets = [float(b.strip()) for b in args.compute_budgets.split(",")]

        experiments = generate_chinchilla_experiments(
            model_names, compute_budgets, slurm_config
        )

        if not experiments:
            print("No experiments generated. Check your model names / budgets.")
            return

        print_scaling_summary(experiments)
        summary = export_experiment_configs(experiments, Path(args.output_dir), slurm_config)

        if args.submit or args.dry_run:
            print("\n" + "=" * 60)
            print("Submitting SLURM Jobs" if not args.dry_run else "Dry-run SLURM Submission")
            print("=" * 60)

            for i, exp_summary in enumerate(summary):
                script_path = exp_summary["script"]
                print(f"\n[{i+1}/{len(summary)}] {exp_summary['name']}")
                job_id = submit_slurm_job(Path(script_path), dry_run=args.dry_run)
                if job_id:
                    exp_summary["job_id"] = job_id
                    exp_summary["status"] = "submitted"
                    print(f"  Submitted: Job ID {job_id}")
                else:
                    exp_summary["status"] = "failed"
                    print(f"  Failed to submit")

            summary_path = Path(args.output_dir) / "experiments_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            submitted = sum(1 for s in summary if s["status"] == "submitted")
            print(f"\nSubmitted {submitted}/{len(summary)} jobs")

    elif args.command == "init_config":
        config = SlurmConfig()
        output_path = args.output

        if output_path.endswith(".yaml") or output_path.endswith(".yml"):
            config.to_yaml(output_path)
        elif output_path.endswith(".json"):
            config.to_json(output_path)
        else:
            raise ValueError("Output file must end with .yaml, .yml, or .json")

        print(f"Template SLURM config saved to: {output_path}")
        print("Edit this file to match your cluster configuration.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
