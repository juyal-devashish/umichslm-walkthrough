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

import warnings
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
        """Active (non-embedding, non-unembedding) parameters — drives D = C/6N.

        Counted from the *instantiated* model rather than a shape formula, because
        the formula below is blind to MLA, MoD routers, global tokens, DeltaNet
        gates, MTP heads and PLE projections. Getting this wrong silently mis-sets
        the token horizon: qwen3_5_like was under-counted by 8.8% and swa_globals
        by 2.9%, so both were being over-fed tokens for their IsoFLOP budget.

        Pure *lookup* tables (input embeddings, PLE) and ``lm_head`` are excluded,
        preserving the original convention. That exclusion is not a fudge: a lookup
        costs no matmul, which is exactly why Gemma-3n can carry 115M of PLE weight
        at ~zero FLOPs and report an "E2B / 2B-effective" parameter count. Use
        :attr:`total_params_real` when you need capacity rather than compute.

        Falls back to the analytic formula if the model cannot be instantiated.
        """
        counted = self._count_real_params()
        if counted is not None:
            return counted["active"]
        return self._num_params_formula

    def _count_real_params(self) -> Optional[Dict[str, int]]:
        """Instantiate the preset and count params by role. Cached per instance."""
        if getattr(self, "_param_cache", None) is not None:
            return self._param_cache
        try:
            from umich_slm.configs.presets import get_umichslm_config
            from umich_slm.modeling_umichslm import UMichSLMForCausalLM

            model = UMichSLMForCausalLM(get_umichslm_config(self.name))
        except Exception as exc:
            # Falling back to the analytic formula is NOT a harmless degradation:
            # the formula is blind to MLA / MoD / globals / DeltaNet gates, so N
            # comes out low and D = C/(6N) comes out proportionally high — the run
            # is then trained at the wrong token horizon, silently. Observed: a bare
            # `python3 scripts/scaling_laws.py` (no PYTHONPATH=src) makes the import
            # fail, and swa_globals_25m is counted as 19.4M instead of 22.5M — a 16%
            # over-long token budget that nothing downstream would ever flag.
            # Swallow it, but never quietly.
            warnings.warn(
                f"{self.name}: could not instantiate the model to count parameters "
                f"({type(exc).__name__}: {exc}). Falling back to the analytic formula, "
                "which UNDERCOUNTS MLA/MoD/globals/DeltaNet and will therefore set the "
                "IsoFLOP token horizon D = C/(6N) TOO HIGH. Run with PYTHONPATH=src (or "
                "pip install -e .) before trusting this manifest.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._param_cache = None
            return None

        total = lookup = head = 0
        for pname, p in model.named_parameters():
            n = p.numel()
            total += n
            # Pure lookup tables: the input embedding and any per-layer embedding
            # table. PLE's *projections* (ple_dim -> hidden) are real matmuls and
            # stay in the active count.
            is_lookup = "embed_tokens" in pname or (
                "per_layer_embeddings" in pname and "projs" not in pname
            )
            if is_lookup:
                lookup += n
            elif "lm_head" in pname:
                head += n

        # Dynamic-compute adjustment: MoD-routed layers bill at routing capacity
        # (FFN-only under mod_mlp_only), recurrent-depth core layers bill
        # E[iterations] times, weight-shared slots bill per execution. Zero for
        # dense archs. This is what keeps the IsoFLOP horizon honest — billing
        # MoD as dense over-charges it and under-feeds it tokens.
        from umich_slm.compute_accounting import model_compute_param_delta

        compute_delta = model_compute_param_delta(model.model, model.config)
        del model
        self._param_cache = {
            "total": total,
            "lookup": lookup,
            "lm_head": head,
            "active": total - lookup - head,
            "compute": int(round(total - lookup - head + compute_delta)),
        }
        return self._param_cache

    @property
    def num_params_compute(self) -> int:
        """Per-token *compute* params — the N in the IsoFLOP horizon D = C/(6N).

        Equals :attr:`num_params` for dense archs. For dynamic-compute archs
        (MoD, MoD-MLP, recurrent depth) it reflects the expected per-token cost:
        routed layers discounted by capacity, looped layers multiplied by
        E[iterations]. Capacity-axis analyses (the L(N,D) fit's N) should keep
        using :attr:`num_params` / :attr:`total_params_real`; only the token
        budget and FLOP bookkeeping use this. NOTE: this changed the horizon for
        the pre-existing ``mod`` and ``swa_globals*`` arms too — pre-fix screen
        rungs are not comparable to post-fix rungs.
        """
        counted = self._count_real_params()
        if counted is not None:
            return counted["compute"]
        return self.num_params

    @property
    def total_params_real(self) -> int:
        """Every trainable parameter, including lookup tables and lm_head.

        This is the number that matters for a *parameter* budget (the 1B target)
        and for the L(N,D) fit's capacity axis — gemma_25m is 167M here versus a
        22.8M active count, and that gap is the whole Gemma-3n trade.
        """
        counted = self._count_real_params()
        if counted is not None:
            return counted["total"]
        return self._num_params_formula + 56000 * self.hidden_size

    @property
    def _num_params_formula(self) -> int:
        """Legacy analytic estimate — fallback only. See :attr:`num_params`.

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
        vocab_size = 56000  # SuperBPE production tokenizer (umich_slm.tokenizer_defaults)
        embedding_params = vocab_size * self.hidden_size
        return self.num_params + embedding_params

    def flops_per_token(self) -> int:
        """Estimate FLOPs per token for training (forward + backward).

        Uses compute params so dynamic-compute archs (MoD/recurrent depth) are
        billed by what they actually execute per token, not their param count.
        """
        return 6 * self.num_params_compute

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
    # When set, emit the Delphi recipe flags (AdamH/WSD/token-horizon LR) into the
    # training command. dict of the recipe *formula inputs* consumed by
    # pretrain_fineweb.py's --delphi_* args; the runtime recomputes the LR/β₂/ε from
    # them + the token horizon. None => plain path (current behaviour).
    delphi: Optional[Dict[str, Any]] = None

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
    # Delphi-style width-scaling (openathena.ai/blog/delphi)
    "delphi_10m",
    "delphi_25m",
    "delphi_50m",
    "delphi_100m",
)

# Matched-N cross-architecture sweep presets (presets.py SWEEP_ARCHS): one shared
# backbone shape per size, differing only in architecture flags. ``<arch>_<size>``.
ARCHITECTURE_PRESET_NAMES = ARCHITECTURE_PRESET_NAMES + tuple(
    f"{_arch}_{_size}"
    for _arch in _presets.SWEEP_ARCHS
    for _size in ("10m", "25m", "50m", "100m")
)


def _build_architecture_model_configs() -> Dict[str, ModelConfig]:
    available = set(list_available_sizes())
    missing = [n for n in ARCHITECTURE_PRESET_NAMES if n not in available]
    if missing:
        raise RuntimeError(
            "scaling_laws.py ARCHITECTURE_PRESET_NAMES includes presets not in presets.py: "
            + ", ".join(missing)
        )
    configs: Dict[str, ModelConfig] = {}
    for name in ARCHITECTURE_PRESET_NAMES:
        cfg = get_preset_info(name)
        configs[name] = ModelConfig(
            name=name,
            config_dict=cfg,
            total_params=estimate_parameters(name),
            hidden_size=int(cfg.get("hidden_size", 0)),
            num_hidden_layers=int(cfg.get("num_hidden_layers", 0)),
            num_attention_heads=int(cfg.get("num_attention_heads", 0) or 0),
            intermediate_size=int(cfg.get("intermediate_size", 0) or 0),
            num_key_value_heads=cfg.get("num_key_value_heads"),
            hybrid_attention_ratio=cfg.get("hybrid_attention_ratio"),
        )
    return configs


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

# Single ~15B-token natural-mix dataset (DATA_SOURCING_PLAN.md §4). Every run
# subsets its D = C/(6N) tokens from this one dataset, so all budgets map to it.
# Built by scripts/prepare_mix_dataset.py (--mix_name pretrain_natural_v1).
_NATURAL_MIX = {
    "dataset": "pretrain_natural_v1_train_15000M",
    "dataset_dir": "/path/to/data",
    "eval_dataset": "pretrain_natural_v1_val_450M",
}
DATASET_CONFIGS = {b: dict(_NATURAL_MIX) for b in (20, 40, 200, 500, 1000, 2000, 5000, 10000, 15000)}

TOKEN_COUNTS_M = [20, 40, 200, 500, 1000]  # Training tokens in millions (chinchilla path)


# ============================================================================
# Hyperparameter Computation
# ============================================================================

def compute_optimal_lr(num_params: int, base_lr: float = 3e-4) -> float:
    """Legacy μP-style LR scaling: η ∝ √(N_ref / N).

    Retained for reproducing pre-Delphi sweeps. New experiments should call
    :func:`compute_horizon_lr` instead, which is token-horizon aware.
    """
    base_params = 125e6
    scale = math.sqrt(base_params / num_params)
    return max(1e-5, min(1e-3, base_lr * scale))


def compute_horizon_lr(
    *,
    num_params: int,
    tokens: float,
    batch: int,
    group: str = "proj",
    eta0_proj: float = 6.30e-3,
    eta0_adam: float = 6.56e-4,
    batch_ref: int = 64,
    token_ref: float = 2.5e9,
    horizon_exp: float = 0.30,
) -> float:
    """Delphi-style token-horizon-aware LR.

    * ``group="proj"`` → η_proj = η₀·√(B/B₀)·(T₀/T)^horizon_exp
    * ``group="adam"`` → η_adam = η₀,adam·√(B·T₀/(B₀·T))

    ``num_params`` is accepted for symmetry but not used — Delphi shows that
    a width-aware η₀ + Hyperball already absorbs the width dependence, so
    the LR depends on (B, T) only. Pass it through for logging.
    """
    del num_params  # See docstring.
    if group == "proj":
        return eta0_proj * math.sqrt(batch / batch_ref) * (token_ref / max(1.0, tokens)) ** horizon_exp
    if group == "adam":
        return eta0_adam * math.sqrt((batch * token_ref) / (batch_ref * max(1.0, tokens)))
    raise ValueError(f"compute_horizon_lr: unknown group {group!r}; expected 'proj' or 'adam'")


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
        slurm_header += f"\n#SBATCH --account=<account>
    if slurm_config.qos:
        slurm_header += f"\n#SBATCH --qos={slurm_config.qos}"
    if slurm_config.mail_account:
        slurm_header += f"\n#SBATCH --mail-user=user@example.com
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

    # Delphi recipe flags: when present, pretrain_fineweb.py's --delphi_recipe path
    # OVERRIDES --learning_rate/--optimizer/--lr_scheduler and recomputes the
    # token-horizon LR (+ AdamH/WSD/β₂/ε) from these formula inputs at runtime.
    delphi_flags = ""
    if exp.delphi:
        r = exp.delphi
        delphi_flags = (
            " \\\n  --delphi_recipe"
            f" \\\n  --delphi_optimizer {r['optimizer']}"
            f" \\\n  --delphi_schedule {r['schedule']}"
            f" \\\n  --delphi_warmup_frac {r['warmup_frac']}"
            f" \\\n  --delphi_decay_frac {r['decay_frac']}"
            f" \\\n  --delphi_eta0_proj {r['eta0_proj']:.6e}"
            f" \\\n  --delphi_eta0_adam {r['eta0_adam']:.6e}"
            f" \\\n  --delphi_batch_ref {int(r['batch_ref'])}"
            f" \\\n  --delphi_token_ref {r['token_ref']:.6e}"
            f" \\\n  --delphi_horizon_exp {r['horizon_exp']}"
            f" \\\n  --delphi_epsilon0 {r['epsilon0']:.6e}"
            f" \\\n  --delphi_beta2_max {r['beta2_max']}"
            f" \\\n  --delphi_beta2_min {r['beta2_min']}"
            f" \\\n  --delphi_grad_clip {r['grad_clip']}"
        )

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
  --do_eval --eval_dataset {exp.eval_dataset} --eval_steps 200{delphi_flags}
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
# Delphi-style sweep generator (openathena.ai/blog/delphi)
# ============================================================================

DELPHI_DEFAULT_ARCHS: tuple[str, ...] = (
    "vanilla",        # vanilla dense transformer: full MHA, RoPE/RMSNorm/SwiGLU
    "qwen3_like",     # dense GQA softmax baseline (Qwen3-style)
    "qwen3_5_like",   # 3:1 Gated DeltaNet / attention hybrid (Qwen3.5-style)
    "mod",            # plain Mixture-of-Depths
    "swa_globals",    # MoD + sliding-window attention + persistent globals (PRIMARY)
    "gemma",          # Gemma-3n/4 style: alt local/global SWA + PLE + MatFormer
    # Depth axis on the primary arch. Block-atomic MoD needs L = 4*(R+2) to expose
    # R routed (skippable) blocks between dense I/O blocks, so depth *is* the
    # "how much can MoD skip" knob — and the 1B target is far deeper than the
    # screen. These are matched-N, deeper-and-narrower variants of swa_globals:
    #   swa_globals        L=12 -> 1 routed block
    #   swa_globals_deep   L=16 -> 2 routed blocks
    #   swa_globals_deeper L=20 -> 3 routed blocks
    "swa_globals_deep",
    "swa_globals_deeper",
)

# Size suffix used for delphi sweeps. Maps shorthand → preset suffix.
DELPHI_SIZE_LABELS: tuple[str, ...] = ("10m", "25m", "50m", "100m")

# Default IsoFLOP budgets in the screening pass (4 budgets so each size lands
# in a different IsoFLOP slice). Held-out lives one decade above.
DELPHI_SCREEN_BUDGETS: tuple[float, ...] = (1e17, 3e17, 1e18, 3e18)
DELPHI_HOLDOUT_BUDGET: float = 1e19


def _delphi_preset_name(arch: str, size_label: str) -> Optional[str]:
    """Map (arch, size) to a preset registered in MODEL_CONFIGS.

    Mapping rules
    -------------
    * Exact matched-N preset ``<arch>_<size>`` (delphi_10m, swa_globals_50m,
      gemma_100m, qwen3_like_25m, …) is preferred when it exists. These share a
      common backbone shape per size so the cross-architecture comparison is
      param-matched (see presets.py SWEEP_ARCHS).
    * ``delphi`` → ``delphi_<size>``.
    * Legacy fallback for older families: 10m→tiny, 25m/50m→xsmall, 100m→small.
      The point estimate uses the preset's actual param count, not the label.
    """
    # Prefer the exact matched-N sweep preset (covers delphi + the 5 sweep archs).
    exact = f"{arch}_{size_label}"
    if exact in MODEL_CONFIGS:
        return exact
    if arch == "delphi":
        return f"delphi_{size_label}"
    size_map = {
        "10m": "tiny",
        "25m": "xsmall",
        "50m": "xsmall",
        "100m": "small",
    }
    suffix = size_map.get(size_label)
    if suffix is None:
        return None
    candidate = f"{arch}_{suffix}"
    if candidate in MODEL_CONFIGS:
        return candidate
    return None


def _delphi_batch_size_for(num_params: int, arch: str = "") -> tuple[int, int]:
    """Return (per_device_batch, grad_accum) targeting global batch ≈ 64."""
    # Conservative ladder; the Delphi recipe is batch-aware via the LR so the
    # exact split between micro-batch and grad-accum doesn't change the LR.
    # Every branch below holds global batch at 64, which keeps the LR / β₂ / ε
    # rules (and hence cross-arch comparability) identical.
    if "mtp" in arch:
        # MTP materialises a SECOND full-vocab logits tensor, and the CE upcasts
        # it to fp32: 32 x 2048 x 56000 x 4B = 14.7 GB, which OOMs an A6000 on the
        # backward. The main head dodges this via chunked CE; the MTP head does
        # not. A quarter-size micro-batch brings that tensor to ~3.7 GB.
        return 8, 8
    if "moda" in arch:
        # MoDA extends every layer's KV with up to 4 depth entries (5x KV length)
        # and runs SDPA with an explicit additive mask, which forces the
        # math/mem-inefficient kernels: attention scores scale with B*H*T*(5T).
        # A quarter-size micro-batch keeps that under the A6000's 48 GB.
        return 8, 8
    if arch.startswith("mop"):
        # MoP's realized implementation runs every lane densely (bitwise equal
        # to gathering, see modules/pathways.py) — at [0,2,4] lanes that's ~1.8x
        # the algorithmic compute/activations, plus a full extra causal
        # cross-attention map per lane block. Half-size micro-batch for slack.
        # (Covers the whole family: mop, mop_dense, mop_kv, mop_nested, mop_rec.)
        return 16, 4
    if num_params < 30e6:
        return 32, 2  # 64 global
    if num_params < 80e6:
        return 16, 4
    return 8, 8


def generate_delphi_experiments(
    *,
    archs: List[str],
    sizes: List[str],
    budgets: List[float],
    seeds: List[int],
    stage_archs: List[str],
    pass_name: str,
    slurm_config: Optional["SlurmConfig"] = None,
    recipe_overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build the list of run specs for a Delphi-style sweep.

    Returns a list of dicts (one per run) that the caller writes into both a
    sweep manifest and into individual SLURM scripts. Each entry carries:

    * ``arch``, ``size_label``, ``preset_name``
    * ``params``, ``tokens``, ``flops``
    * ``batch_size``, ``grad_accum``, ``learning_rate``, ``learning_rate_adam``
    * ``schedule``, ``optimizer``, ``grad_clip``, ``beta2``, ``epsilon``
    * ``seed``, ``stages`` (list of stage dicts for staged variants),
      ``pass`` (``"screen"`` or ``"holdout"``), ``held_out`` (bool).
    """
    overrides = dict(recipe_overrides or {})
    eta0_proj = float(overrides.get("eta0_proj", 6.30e-3))
    eta0_adam = float(overrides.get("eta0_adam", 6.56e-4))
    batch_ref = int(overrides.get("batch_ref", 64))
    token_ref = float(overrides.get("token_ref", 2.5e9))
    horizon_exp = float(overrides.get("horizon_exp", 0.30))
    grad_clip = float(overrides.get("grad_clip", 0.1))
    schedule = str(overrides.get("schedule", "wsd"))
    optimizer = str(overrides.get("optimizer", "adamh"))
    warmup_frac = float(overrides.get("warmup_frac", 0.10))
    decay_frac = float(overrides.get("decay_frac", 0.20))
    epsilon0 = float(overrides.get("epsilon0", 1.85e-8))
    beta2_max = float(overrides.get("beta2_max", 0.9999))
    beta2_min = float(overrides.get("beta2_min", 0.9))

    runs: List[Dict[str, Any]] = []
    for arch in archs:
        for size in sizes:
            preset = _delphi_preset_name(arch, size)
            if preset is None or preset not in MODEL_CONFIGS:
                print(f"  !! delphi_sweep: no preset for ({arch}, {size}); skipping")
                continue
            mc = MODEL_CONFIGS[preset]
            n = mc.num_params
            # The token horizon is set from *compute* params: dynamic-compute
            # archs (MoD / MoD-MLP / recurrent depth) execute fewer (or more)
            # params per token than they own, and billing them as dense
            # mis-feeds them tokens for the IsoFLOP budget. n (capacity) is
            # still what gets reported to the L(N,D) fit.
            n_compute = mc.num_params_compute
            for c in budgets:
                d = c / (6.0 * n_compute)
                if d < 1e6:
                    continue
                per_dev_b, accum = _delphi_batch_size_for(n, arch)
                global_batch = per_dev_b * accum
                lr_proj = compute_horizon_lr(
                    num_params=n, tokens=d, batch=global_batch, group="proj",
                    eta0_proj=eta0_proj, eta0_adam=eta0_adam,
                    batch_ref=batch_ref, token_ref=token_ref, horizon_exp=horizon_exp,
                )
                lr_adam = compute_horizon_lr(
                    num_params=n, tokens=d, batch=global_batch, group="adam",
                    eta0_proj=eta0_proj, eta0_adam=eta0_adam,
                    batch_ref=batch_ref, token_ref=token_ref, horizon_exp=horizon_exp,
                )
                # β₂ clip rule and ε rule
                beta2 = min(max(beta2_max ** (global_batch / batch_ref), beta2_min), beta2_max)
                eps = epsilon0 * math.sqrt((batch_ref * d) / (global_batch * token_ref))
                # Determine staged variants for this (arch, size).
                stage_variants = ["single"]
                if arch in stage_archs and size in ("50m", "100m"):
                    stage_variants.append("staged")
                for variant in stage_variants:
                    for seed in seeds:
                        run = {
                            "arch": arch,
                            "size_label": size,
                            "preset_name": preset,
                            "variant": variant,
                            "params": int(n),
                            "params_compute": int(n_compute),
                            "tokens": int(d),
                            "flops": c,
                            "batch_size": per_dev_b,
                            "grad_accum": accum,
                            "global_batch": global_batch,
                            "learning_rate": lr_proj,
                            "learning_rate_adam": lr_adam,
                            "schedule": schedule,
                            "optimizer": optimizer,
                            "grad_clip": grad_clip,
                            "beta2": beta2,
                            "epsilon": eps,
                            "warmup_frac": warmup_frac,
                            "decay_frac": decay_frac,
                            "seed": int(seed),
                            "pass": pass_name,
                            "held_out": pass_name == "holdout",
                            # Recipe *formula inputs* consumed by pretrain_fineweb.py's
                            # --delphi_* args (it recomputes LR/β₂/ε from these + the
                            # token horizon at runtime). Emitted into the training cmd
                            # by generate_slurm_script so the run actually uses AdamH/WSD.
                            "recipe": {
                                "optimizer": optimizer,
                                "schedule": schedule,
                                "warmup_frac": warmup_frac,
                                "decay_frac": decay_frac,
                                "eta0_proj": eta0_proj,
                                "eta0_adam": eta0_adam,
                                "batch_ref": batch_ref,
                                "token_ref": token_ref,
                                "horizon_exp": horizon_exp,
                                "epsilon0": epsilon0,
                                "beta2_max": beta2_max,
                                "beta2_min": beta2_min,
                                "grad_clip": grad_clip,
                            },
                        }
                        if variant == "staged":
                            # Build a two-stage plan: train smaller→grown.
                            parent_label = "10m" if size == "50m" else ("25m" if size == "100m" else None)
                            if parent_label is None:
                                continue
                            parent_preset = _delphi_preset_name(arch, parent_label)
                            if parent_preset is None:
                                continue
                            parent_n = MODEL_CONFIGS[parent_preset].num_params
                            # 30 % of tokens in stage 1, 70 % in stage 2.
                            d1 = int(d * 0.3)
                            d2 = int(d * 0.7)
                            run["stages"] = [
                                {"params": parent_n, "tokens": d1, "arch": arch, "preset": parent_preset},
                                {"params": int(n), "tokens": d2, "arch": arch, "preset": preset},
                            ]
                        runs.append(run)
    return runs


def write_delphi_manifest(
    runs: List[Dict[str, Any]], output_dir: Path, *, append: bool = True
) -> Path:
    """Write/append the sweep manifest at ``<output_dir>/sweep_manifest.json``.

    Append mode merges by ``run_id`` (preset+variant+seed+flops+pass). On
    conflict the new entry overwrites; this lets re-runs update metadata
    (e.g. a job ID) without losing other entries.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "sweep_manifest.json"
    existing: Dict[str, Dict[str, Any]] = {}
    if append and path.exists():
        try:
            with open(path) as f:
                blob = json.load(f)
            for entry in blob.get("runs", []):
                existing[entry["run_id"]] = entry
        except (json.JSONDecodeError, KeyError):
            existing = {}
    for r in runs:
        run_id = (
            f"{r['arch']}-{r['preset_name']}-{r['variant']}-c{r['flops']:.0e}-s{r['seed']}-{r['pass']}"
        )
        r["run_id"] = run_id
        existing[run_id] = r
    blob = {
        "schema_version": 1,
        "runs": sorted(existing.values(), key=lambda x: (x["arch"], x["params"], x["flops"], x["seed"])),
    }
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    return path


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
        help="Uniqname for Slurm mail (sets --mail-user=user@example.com empty string disables",
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

    # Delphi-style sweep command
    delphi_parser = subparsers.add_parser(
        "delphi_sweep",
        help="Generate Delphi-style IsoFLOP sweep (screening or held-out pass)",
    )
    delphi_parser.add_argument("--output_dir", type=str, required=True,
                               help="Output directory; SLURM scripts + sweep_manifest.json land here")
    delphi_parser.add_argument(
        "--pass",
        dest="delphi_pass",
        choices=("screen", "holdout"),
        default="screen",
        help="screen = full IsoFLOP grid (1 seed); holdout = held-out budget only (--seeds)",
    )
    delphi_parser.add_argument(
        "--archs", type=str, default=",".join(DELPHI_DEFAULT_ARCHS),
        help="Comma-separated architecture families to sweep",
    )
    delphi_parser.add_argument(
        "--sizes", type=str, default=",".join(DELPHI_SIZE_LABELS),
        help="Comma-separated size labels (10m,25m,50m,100m)",
    )
    delphi_parser.add_argument(
        "--budgets", type=str,
        default=",".join(f"{b:.0e}" for b in DELPHI_SCREEN_BUDGETS),
        help="Screen-pass compute budgets in FLOPs",
    )
    delphi_parser.add_argument(
        "--holdout_budget", type=float, default=DELPHI_HOLDOUT_BUDGET,
        help="Held-out compute budget (FLOPs) for --pass holdout",
    )
    delphi_parser.add_argument(
        "--seeds", type=str, default="0",
        help="Comma-separated seeds (default '0'; holdout pass should use '0,1,2')",
    )
    delphi_parser.add_argument(
        "--stage_archs", type=str, default="swa_globals,mod",
        help="Comma-separated archs to ALSO emit staged (layer-grown) variants for at "
             "50m/100m. Each such arch is then screened both flat (stacking off) and "
             "staged (stacking on) — the FLOPs-saved-by-stacking comparison "
             "(IMPLEMENTATION_PLAN_SWA.md §3). Default is the primary swa_globals + mod.",
    )
    delphi_parser.add_argument("--dataset", type=str, default=None,
                               help="Dataset name (overrides config)")
    delphi_parser.add_argument("--dataset_dir", type=str, default=None,
                               help="Dataset directory (overrides config)")
    delphi_parser.add_argument("--dry_run", action="store_true",
                               help="Print sweep plan + write manifest without writing SLURM scripts")
    # Per-knob recipe flags
    recipe = delphi_parser.add_argument_group("Recipe knobs (Delphi defaults)")
    recipe.add_argument("--lr_eta0_proj", type=float, default=6.30e-3)
    recipe.add_argument("--lr_eta0_adam", type=float, default=6.56e-4)
    recipe.add_argument("--lr_batch_ref", type=int, default=64)
    recipe.add_argument("--lr_token_ref", type=float, default=2.5e9)
    recipe.add_argument("--lr_horizon_exp", type=float, default=0.30)
    recipe.add_argument("--schedule", type=str, choices=("wsd", "cosine", "linear"), default="wsd")
    recipe.add_argument("--warmup_frac", type=float, default=0.10)
    recipe.add_argument("--decay_frac", type=float, default=0.20)
    recipe.add_argument(
        "--optimizer",
        type=str,
        choices=("adamh", "adamw", "muonh"),
        default="adamh",
        help="adamh (default) = Adam+Hyperball on all proj weights; muonh = full "
        "Delphi hybrid (MuonH hidden + AdamH head + AdamW embed/scalar), single-GPU "
        "only; adamw = plain baseline.",
    )
    recipe.add_argument("--gradient_clip", type=float, default=0.1)
    recipe.add_argument("--epsilon0", type=float, default=1.85e-8)
    recipe.add_argument("--beta2_max", type=float, default=0.9999)
    recipe.add_argument("--beta2_min", type=float, default=0.9)
    add_slurm_args(delphi_parser)

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

    elif args.command == "delphi_sweep":
        slurm_config = load_slurm_config(args)
        if args.dataset:
            slurm_config.dataset = args.dataset
        if args.dataset_dir:
            slurm_config.dataset_dir = args.dataset_dir

        archs = [a.strip() for a in args.archs.split(",") if a.strip()]
        sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        stage_archs = [a.strip() for a in args.stage_archs.split(",") if a.strip()]
        budgets = (
            [args.holdout_budget]
            if args.delphi_pass == "holdout"
            else [float(b.strip()) for b in args.budgets.split(",") if b.strip()]
        )

        recipe_overrides = {
            "eta0_proj": args.lr_eta0_proj,
            "eta0_adam": args.lr_eta0_adam,
            "batch_ref": args.lr_batch_ref,
            "token_ref": args.lr_token_ref,
            "horizon_exp": args.lr_horizon_exp,
            "schedule": args.schedule,
            "warmup_frac": args.warmup_frac,
            "decay_frac": args.decay_frac,
            "optimizer": args.optimizer,
            "grad_clip": args.gradient_clip,
            "epsilon0": args.epsilon0,
            "beta2_max": args.beta2_max,
            "beta2_min": args.beta2_min,
        }

        runs = generate_delphi_experiments(
            archs=archs, sizes=sizes, budgets=budgets, seeds=seeds,
            stage_archs=stage_archs, pass_name=args.delphi_pass,
            slurm_config=slurm_config, recipe_overrides=recipe_overrides,
        )
        if not runs:
            print("delphi_sweep: no runs generated.")
            return

        # Print a summary table
        print("\n" + "=" * 110)
        print(f"Delphi sweep — pass={args.delphi_pass}  archs={archs}  sizes={sizes}")
        print("=" * 110)
        print(f"{'arch':<14} {'size':<6} {'variant':<8} {'N':>10} {'D':>10} {'C (FLOPs)':>12} {'B':>4} {'lr_proj':>10} {'lr_adam':>10} {'seed':>5}")
        print("-" * 110)
        for r in runs:
            print(
                f"{r['arch']:<14} {r['size_label']:<6} {r['variant']:<8} "
                f"{r['params']/1e6:>9.1f}M {r['tokens']/1e6:>9.1f}M "
                f"{r['flops']:>12.2e} {r['global_batch']:>4} "
                f"{r['learning_rate']:>10.2e} {r['learning_rate_adam']:>10.2e} "
                f"{r['seed']:>5}"
            )
        print("-" * 110)
        print(f"Total runs: {len(runs)}\n")

        output_dir = Path(args.output_dir)
        manifest_path = write_delphi_manifest(runs, output_dir)
        print(f"Manifest written to: {manifest_path}")

        if args.dry_run:
            print("[--dry_run] skipping SLURM script generation.")
            return

        # Generate SLURM scripts as ScalingExperiment objects (single-stage only;
        # staged variants get a sibling .stages.json next to the script that the
        # training stack reads to drive layer growth).
        scripts_dir = output_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        for r in runs:
            mc = MODEL_CONFIGS[r["preset_name"]]
            exp = ScalingExperiment(
                model_config=mc,
                num_tokens=int(r["tokens"]),
                batch_size=int(r["batch_size"]),
                gradient_accumulation_steps=int(r["grad_accum"]),
                learning_rate=float(r["learning_rate"]),
                dataset=slurm_config.dataset,
                dataset_dir=slurm_config.dataset_dir,
                eval_dataset=slurm_config.eval_dataset,
                partition=GPU_BY_SIZE.get(_chinchilla_size_suffix(r["preset_name"]), {}).get("partition"),
                delphi=r.get("recipe"),
            )
            script_path = generate_slurm_script(exp, slurm_config, scripts_dir)
            # Sidecar JSON with recipe + staged-plan metadata
            sidecar = {k: v for k, v in r.items() if k != "preset_name"}
            sidecar["preset_name"] = r["preset_name"]
            with open(script_path.with_suffix(".recipe.json"), "w") as f:
                json.dump(sidecar, f, indent=2)
        print(f"Wrote {len(runs)} SLURM scripts + recipe sidecars in {scripts_dir}")

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
