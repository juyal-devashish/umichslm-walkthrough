#!/usr/bin/env python3
"""Pretraining script for UMichSLM using LLaMA Factory.

This script:
1. Creates a UMichSLM model with specified configuration
2. Saves the model/config for LLaMA Factory to load
3. Generates a LLaMA Factory YAML config
4. Runs pretraining via LLaMA Factory CLI

Example usage:
    # Basic pretraining with small model
    python scripts/pretrain.py --size small --output_dir ./outputs/pretrain-small

    # With MoD enabled
    python scripts/pretrain.py --size small --use_mod --output_dir ./outputs/pretrain-mod

    # With PoPE instead of RoPE
    python scripts/pretrain.py --size small --position_embedding_type pope

    # Full ablation run
    python scripts/pretrain.py --size small --use_mod --use_mtp --wandb_project umichslm
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


def get_run_name(args: argparse.Namespace) -> str:
    """Generate a descriptive run name including the date."""
    parts = [args.size]

    # Position embedding type
    if args.position_embedding_type != "rope":
        parts.append(args.position_embedding_type)
    if args.rope_type != "default":
        parts.append(f"{args.rope_type}{int(args.rope_scaling_factor)}x")

    # Features
    if args.use_mod:
        parts.append(f"mod{int(args.mod_capacity * 100)}")
    if args.use_mtp:
        parts.append(f"mtp{args.mtp_num_tokens}")
    if args.use_moe:
        parts.append(f"moe{args.num_experts}x{args.num_experts_per_tok}")
    if args.sliding_window:
        parts.append(f"sw{args.sliding_window}")
    if args.use_hymba:
        parts.append("hymba")
    if args.use_griffin:
        parts.append("griffin")
    if args.use_block_weight_sharing:
        parts.append("ls")
    if args.num_meta_tokens > 0:
        parts.append(f"meta{args.num_meta_tokens}")

    # Add date
    date_str = datetime.now().strftime("%Y%m%d-%H%M")
    parts.append(date_str)

    return "_".join(parts)


def create_model_config(args: argparse.Namespace) -> dict:
    """Create model configuration dictionary."""
    from umich_slm.configs.presets import UMICHSLM_CONFIGS

    # Start with preset config
    if args.size not in UMICHSLM_CONFIGS:
        raise ValueError(f"Unknown size: {args.size}. Available: {list(UMICHSLM_CONFIGS.keys())}")

    config = UMICHSLM_CONFIGS[args.size].copy()

    # Position embedding
    config["position_embedding_type"] = args.position_embedding_type
    config["rope_theta"] = args.rope_theta

    # Mixture of Depths
    config["use_mod"] = args.use_mod
    if args.use_mod:
        config["mod_capacity"] = args.mod_capacity
        config["mod_every_n_layers"] = args.mod_every_n_layers
        config["mod_aux_loss_weight"] = args.mod_aux_loss_weight

    # Multi-Token Prediction
    config["use_mtp"] = args.use_mtp
    if args.use_mtp:
        config["mtp_num_heads"] = args.mtp_num_tokens
        config["mtp_loss_weight"] = args.mtp_loss_weight

    # Token-Superposition Training (TST). Flags are saved on the init config so
    # downstream trainers (LLaMA Factory, HF Trainer) can attach TSTTrainerCallback.
    config["tst_enabled"] = bool(getattr(args, "tst_enabled", False))
    if config["tst_enabled"]:
        config["tst_bag_size"] = int(args.tst_bag_size)
        config["tst_recovery_ratio"] = float(args.tst_recovery_ratio)
        config["tst_loss_weighting"] = args.tst_loss_weighting

    # Mixture of Experts
    config["use_moe"] = args.use_moe
    if args.use_moe:
        config["num_experts"] = args.num_experts
        config["num_experts_per_tok"] = args.num_experts_per_tok
        config["num_shared_experts"] = args.num_shared_experts

    # Sliding Window Attention
    config["sliding_window"] = args.sliding_window
    if args.sliding_window:
        config["global_attention_every_n_layers"] = args.global_attention_every_n_layers

    # DRoPE (Dropped Rotary Position Embeddings)
    config["use_drope"] = args.use_drope
    if args.use_drope:
        config["drope_drop_fraction"] = args.drope_drop_fraction

    # Hymba (Hybrid Mamba-Attention)
    config["use_hymba"] = args.use_hymba
    if args.use_hymba:
        config["hymba_num_meta_tokens"] = args.hymba_num_meta_tokens
        config["hymba_ssm_state_size"] = args.hymba_ssm_state_size
        config["hymba_ssm_expand"] = args.hymba_ssm_expand
        config["hymba_share_kv_layers"] = args.hymba_share_kv_layers

    # Meta tokens (available for all architectures)
    config["num_meta_tokens"] = args.num_meta_tokens

    # Training settings
    config["attention_dropout"] = args.attention_dropout
    config["tie_word_embeddings"] = args.tie_word_embeddings

    # MobileLLM block-wise weight sharing
    config["use_block_weight_sharing"] = args.use_block_weight_sharing
    if args.use_block_weight_sharing:
        config["weight_sharing_block_size"] = args.weight_sharing_block_size

    # Griffin (RG-LRU hybrid)
    config["use_griffin"] = args.use_griffin
    if args.use_griffin:
        config["griffin_pattern"] = [p.strip() for p in args.griffin_pattern.split(",")]
        if args.griffin_num_heads is not None:
            config["griffin_num_heads"] = args.griffin_num_heads

    # YaRN RoPE scaling
    if args.rope_type != "default":
        config["rope_scaling"] = {"type": args.rope_type, "factor": args.rope_scaling_factor}
        if args.rope_type == "yarn":
            config["rope_scaling"]["beta_fast"] = args.yarn_beta_fast
            config["rope_scaling"]["beta_slow"] = args.yarn_beta_slow
            config["rope_scaling"]["mscale"] = args.yarn_mscale

    return config


def save_model_init(config_dict: dict, output_path: Path, tokenizer_path: Optional[str] = None) -> None:
    """Save initialized model for LLaMA Factory to load.

    Args:
        config_dict: Model configuration dictionary
        output_path: Path to save the model
        tokenizer_path: Optional path to an existing tokenizer to copy. If None,
            creates a new LlamaTokenizer with the model's vocab_size.
    """
    from umich_slm import UMichSLMConfig, UMichSLMForCausalLM
    from transformers import AutoTokenizer
    from transformers.models.llama import LlamaTokenizer

    output_path.mkdir(parents=True, exist_ok=True)

    # Load tokenizer first to get the correct vocab_size
    if tokenizer_path:
        # Copy tokenizer from existing path
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    else:
        # Default to the project SuperBPE-56000 tokenizer (umich_slm.tokenizer_defaults).
        from umich_slm.tokenizer_defaults import get_tokenizer_path
        default_tokenizer_path = get_tokenizer_path()
        print(f"Loading tokenizer from {default_tokenizer_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                default_tokenizer_path,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error: Could not load tokenizer from {default_tokenizer_path}: {e}")
            print("Please provide --tokenizer_path with a valid LLaMA tokenizer")
            raise

    # Update config_dict to match tokenizer vocab_size
    if tokenizer.vocab_size != config_dict.get("vocab_size", 32000):
        print(f"Updating model vocab_size from {config_dict.get('vocab_size', 32000)} to {tokenizer.vocab_size} to match tokenizer")
        config_dict["vocab_size"] = tokenizer.vocab_size

    # Create config and model with correct vocab_size
    config = UMichSLMConfig(**config_dict)
    model = UMichSLMForCausalLM(config)

    # Save model and config
    model.save_pretrained(output_path)
    print(f"Model saved to {output_path}")

    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    print(f"Tokenizer saved to {output_path}")

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # CRITICAL: Explicitly delete model and clear CUDA cache to free GPU memory
    # Without this, the model stays in memory when LLaMA Factory loads its own copy,
    # causing OOM even for small models
    import torch
    del model
    del config
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared after model save")


def generate_llamafactory_config(
    args: argparse.Namespace,
    model_path: Path,
    run_name: str,
) -> dict:
    """Generate LLaMA Factory YAML configuration."""
    config = {
        # Model Configuration
        "model_name_or_path": str(model_path),
        "trust_remote_code": True,

        # Data Configuration
        "dataset": args.dataset,
        # For pretraining, we don't need a chat template
        # "template": "default",  # Not needed for pt stage
        "cutoff_len": args.cutoff_len,
        "preprocessing_num_workers": args.num_workers,
        # Point to dataset_info.json location (use absolute path for reliability)
        "dataset_dir": str(Path(__file__).parent.parent.absolute() / "configs" / "llama_factory"),

        # Training Stage
        "stage": "pt",  # Pretraining
        "do_train": True,
        "finetuning_type": "full",

        # Output Configuration
        "output_dir": str(Path(args.output_dir) / run_name),
        "logging_dir": str(Path(args.output_dir) / run_name / "logs"),
        "overwrite_output_dir": True,
        "run_name": run_name,

        # Training Hyperparameters
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_epochs,
        "max_steps": args.max_steps if args.max_steps > 0 else -1,
        "lr_scheduler_type": args.lr_scheduler,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,

        # Optimization
        "bf16": args.bf16,
        "fp16": not args.bf16 and args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "flash_attn": "fa2" if args.flash_attn else "disabled",
        "optim": args.optimizer,

        # Logging and Saving
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,

        # Evaluation
        "do_eval": args.do_eval,
        "eval_strategy": "steps" if args.do_eval else "no",

        # Misc
        "seed": args.seed,
        "dataloader_num_workers": args.num_workers,
        "dataloader_pin_memory": True,
    }

    # Wandb integration
    if args.wandb_project:
        config["report_to"] = "wandb"
        config["wandb_project"] = args.wandb_project
        if args.wandb_entity:
            config["wandb_entity"] = args.wandb_entity
    else:
        config["report_to"] = "tensorboard"

    # DeepSpeed for distributed training
    if args.deepspeed:
        config["deepspeed"] = args.deepspeed

    # Limit number of samples for debugging
    if args.max_samples:
        config["max_samples"] = args.max_samples

    return config


def register_smollm_dataset():
    """Register SmolLM-corpus dataset with LLaMA Factory.

    LLaMA Factory uses a dataset_info.json file to register datasets.
    This function creates the necessary entries.

    For pretraining (stage=pt), datasets should NOT have formatting specified.
    Just map the text column directly.
    """
    dataset_info = {
        "smollm_cosmopedia": {
            "hf_hub_url": "HuggingFaceTB/smollm-corpus",
            "subset": "cosmopedia-v2",
            "columns": {"prompt": "text"},
        },
        "smollm_fineweb": {
            "hf_hub_url": "HuggingFaceTB/smollm-corpus",
            "subset": "fineweb-edu-dedup",
            "columns": {"prompt": "text"},
        },
        "smollm_python": {
            "hf_hub_url": "HuggingFaceTB/smollm-corpus",
            "subset": "python-edu",
            "columns": {"prompt": "text"},
        },
    }
    return dataset_info


def run_training(config_path: Path, dry_run: bool = False) -> int:
    """Run LLaMA Factory training.

    Uses a Python wrapper that ensures the umich_slm package is registered
    with Transformers before LlamaFactory tries to load the model.
    """
    # Use Python wrapper to ensure model registration
    script_dir = Path(__file__).parent
    wrapper_script = script_dir / "llamafactory_train.py"

    if wrapper_script.exists():
        # Use the wrapper script that registers the model
        cmd = [sys.executable, str(wrapper_script), "train", str(config_path)]
    else:
        # Fallback to direct llamafactory-cli (may fail if model not registered)
        print("⚠ Warning: Wrapper script not found, using llamafactory-cli directly")
        print("  Make sure umich_slm package is installed: pip install -e .")
        cmd = ["llamafactory-cli", "train", str(config_path)]

    if dry_run:
        print(f"Dry run - would execute: {' '.join(cmd)}")
        return 0

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    from umich_slm.configs.presets import UMICHSLM_CONFIGS

    parser = argparse.ArgumentParser(
        description="Pretrain UMichSLM using LLaMA Factory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--size",
        type=str,
        default="small",
        choices=list(UMICHSLM_CONFIGS.keys()),
        help="Model size preset",
    )
    model_group.add_argument(
        "--position_embedding_type",
        type=str,
        default="rope",
        choices=["rope", "pope"],
        help="Position embedding type (RoPE or PoPE)",
    )
    model_group.add_argument(
        "--rope_theta",
        type=float,
        default=10000.0,
        help="Base frequency for RoPE",
    )
    model_group.add_argument(
        "--tie_word_embeddings",
        action="store_true",
        help="Tie input/output embeddings",
    )
    model_group.add_argument(
        "--attention_dropout",
        type=float,
        default=0.0,
        help="Attention dropout rate",
    )

    # Mixture of Depths
    mod_group = parser.add_argument_group("Mixture of Depths")
    mod_group.add_argument(
        "--use_mod",
        action="store_true",
        help="Enable Mixture of Depths",
    )
    mod_group.add_argument(
        "--mod_capacity",
        type=float,
        default=0.125,
        help="Fraction of tokens to process per MoD layer",
    )
    mod_group.add_argument(
        "--mod_every_n_layers",
        type=int,
        default=2,
        help="Apply MoD every N layers",
    )
    mod_group.add_argument(
        "--mod_aux_loss_weight",
        type=float,
        default=0.01,
        help="Weight for MoD auxiliary loss",
    )

    # Multi-Token Prediction
    mtp_group = parser.add_argument_group("Multi-Token Prediction")
    mtp_group.add_argument(
        "--use_mtp",
        action="store_true",
        help="Enable Multi-Token Prediction",
    )
    mtp_group.add_argument(
        "--mtp_num_tokens",
        type=int,
        default=2,
        help="Number of additional tokens to predict beyond the next token (e.g., 1 = predict 2 tokens total, 2 = predict 3 tokens total)",
    )
    mtp_group.add_argument(
        "--mtp_loss_weight",
        type=float,
        default=0.3,
        help="Weight for MTP loss",
    )

    # Token-Superposition Training (Peng et al., arXiv:2605.06546)
    tst_group = parser.add_argument_group("Token-Superposition Training")
    tst_group.add_argument(
        "--tst_enabled",
        action="store_true",
        help="Enable TST phase (i) + recovery. Paper: arXiv:2605.06546.",
    )
    tst_group.add_argument(
        "--tst_bag_size",
        type=int,
        default=6,
        help="TST bag size s (paper: 4-8 dense ≤3B, 16 for 10B-A1B MoE).",
    )
    tst_group.add_argument(
        "--tst_recovery_ratio",
        type=float,
        default=0.3,
        help="Fraction of total steps in TST phase (i) before recovery.",
    )
    tst_group.add_argument(
        "--tst_loss_weighting",
        type=str,
        default="uniform",
        choices=["uniform", "power_law"],
        help="Within-bag weighting for MCE (power_law per Appendix D for s>=8).",
    )

    # Mixture of Experts
    moe_group = parser.add_argument_group("Mixture of Experts")
    moe_group.add_argument(
        "--use_moe",
        action="store_true",
        help="Enable Mixture of Experts",
    )
    moe_group.add_argument(
        "--num_experts",
        type=int,
        default=8,
        help="Number of expert networks",
    )
    moe_group.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=2,
        help="Number of experts per token (top-k)",
    )
    moe_group.add_argument(
        "--num_shared_experts",
        type=int,
        default=0,
        help="Number of shared experts",
    )

    # Sliding Window Attention
    swa_group = parser.add_argument_group("Sliding Window Attention")
    swa_group.add_argument(
        "--sliding_window",
        type=int,
        default=None,
        help="Sliding window size (None for full attention)",
    )
    swa_group.add_argument(
        "--global_attention_every_n_layers",
        type=int,
        default=4,
        help="Use global attention every N layers",
    )

    # Hymba (Hybrid Mamba-Attention)
    hymba_group = parser.add_argument_group("Hymba (Hybrid Mamba-Attention)")
    hymba_group.add_argument(
        "--use_hymba",
        action="store_true",
        help="Enable Hymba hybrid attention-SSM heads",
    )
    hymba_group.add_argument(
        "--hymba_num_meta_tokens",
        type=int,
        default=128,
        help="Number of learnable meta tokens for Hymba",
    )
    hymba_group.add_argument(
        "--hymba_ssm_state_size",
        type=int,
        default=16,
        help="SSM state dimension for Mamba2 in Hymba",
    )
    hymba_group.add_argument(
        "--hymba_ssm_expand",
        type=int,
        default=2,
        help="SSM inner dimension multiplier",
    )
    hymba_group.add_argument(
        "--hymba_share_kv_layers",
        action="store_true",
        help="Enable cross-layer KV cache sharing",
    )

    # Meta tokens (available for all architectures)
    model_group.add_argument(
        "--num_meta_tokens",
        type=int,
        default=0,
        help="Number of learnable meta tokens prepended to input (0 = disabled, works with any architecture)",
    )

    # MobileLLM block-wise weight sharing
    mobilellm_group = parser.add_argument_group("MobileLLM Block-Wise Weight Sharing")
    mobilellm_group.add_argument(
        "--use_block_weight_sharing",
        action="store_true",
        help="Enable block-wise weight sharing (MobileLLM-LS): consecutive layer pairs share weights",
    )
    mobilellm_group.add_argument(
        "--weight_sharing_block_size",
        type=int,
        default=2,
        help="Number of consecutive layers that share weights (default 2 = pairs)",
    )

    # Griffin (RG-LRU hybrid)
    griffin_group = parser.add_argument_group("Griffin (RG-LRU Hybrid Recurrent-Attention)")
    griffin_group.add_argument(
        "--use_griffin",
        action="store_true",
        help="Enable Griffin RG-LRU hybrid architecture",
    )
    griffin_group.add_argument(
        "--griffin_pattern",
        type=str,
        default="rglru,rglru,attn",
        help="Comma-separated repeating layer pattern, e.g. 'rglru,rglru,attn' (paper default)",
    )
    griffin_group.add_argument(
        "--griffin_num_heads",
        type=int,
        default=None,
        help="Number of RG-LRU heads (defaults to num_attention_heads)",
    )

    # YaRN RoPE scaling
    yarn_group = parser.add_argument_group("YaRN RoPE Scaling")
    yarn_group.add_argument(
        "--rope_type",
        type=str,
        default="default",
        choices=["default", "linear", "dynamic", "yarn"],
        help="RoPE scaling type ('yarn' enables YaRN context extension)",
    )
    yarn_group.add_argument(
        "--rope_scaling_factor",
        type=float,
        default=1.0,
        help="Context extension factor (e.g. 4.0 for 4x longer context); must be > 1.0 when rope_type != default",
    )
    yarn_group.add_argument(
        "--yarn_beta_fast",
        type=float,
        default=32.0,
        help="YaRN high-frequency threshold (beta_fast); controls where ramp starts",
    )
    yarn_group.add_argument(
        "--yarn_beta_slow",
        type=float,
        default=1.0,
        help="YaRN low-frequency threshold (beta_slow); controls where ramp ends",
    )
    yarn_group.add_argument(
        "--yarn_mscale",
        type=float,
        default=1.0,
        help="YaRN attention temperature scaling multiplier",
    )

    # DRoPE (Dropped Rotary Position Embeddings)
    drope_group = parser.add_argument_group("DRoPE (Dropped Position Embeddings)")
    drope_group.add_argument(
        "--use_drope",
        action="store_true",
        help="Enable DRoPE training (drop position embeddings after drope_drop_fraction of training)",
    )
    drope_group.add_argument(
        "--drope_drop_fraction",
        type=float,
        default=0.875,
        help="Fraction of training to use position embeddings before dropping (default 0.875 = drop at 87.5%%)",
    )

    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--dataset",
        type=str,
        default="smollm_cosmopedia",
        help="Dataset name (smollm_cosmopedia, smollm_fineweb, smollm_python, or custom)",
    )
    data_group.add_argument(
        "--cutoff_len",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    data_group.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for debugging/testing). If None, uses all samples.",
    )

    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device batch size",
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    train_group.add_argument(
        "--num_epochs",
        type=float,
        default=1.0,
        help="Number of training epochs",
    )
    train_group.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Max training steps (-1 for unlimited)",
    )
    train_group.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="Learning rate scheduler",
    )
    train_group.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Warmup ratio",
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay",
    )
    train_group.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    train_group.add_argument(
        "--optimizer",
        type=str,
        default="adamw_torch",
        help="Optimizer (adamw_torch, adamw_torch_fused, deepspeed)",
    )

    # Precision and optimization
    opt_group = parser.add_argument_group("Precision and Optimization")
    opt_group.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision",
    )
    opt_group.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 precision",
    )
    opt_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing",
    )
    opt_group.add_argument(
        "--flash_attn",
        action="store_true",
        default=True,
        help="Use Flash Attention 2",
    )
    opt_group.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed config file path",
    )

    # Logging and saving
    log_group = parser.add_argument_group("Logging and Saving")
    log_group.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    log_group.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Logging interval (steps)",
    )
    log_group.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Checkpoint save interval (steps)",
    )
    log_group.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Evaluation interval (steps)",
    )
    log_group.add_argument(
        "--save_total_limit",
        type=int,
        default=5,
        help="Maximum number of checkpoints to keep",
    )
    log_group.add_argument(
        "--do_eval",
        action="store_true",
        help="Run evaluation during training",
    )

    # Wandb
    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Wandb project name (enables wandb logging)",
    )
    wandb_group.add_argument(
        "--wandb_entity",
        type=str,
        default="umich-slm",
        help="Wandb entity/team name",
    )
    wandb_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Custom run name (auto-generated if not specified)",
    )

    # Misc
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    misc_group.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable fully deterministic mode (may slow training by 5-15%%)",
    )
    misc_group.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    misc_group.add_argument(
        "--dry_run",
        action="store_true",
        help="Generate configs without running training",
    )
    misc_group.add_argument(
        "--config_only",
        action="store_true",
        help="Only save configs, don't initialize model",
    )
    misc_group.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to an existing tokenizer to copy (e.g., meta-llama/Meta-Llama-3-8B). "
             "If not provided, will use LLaMA 3 tokenizer by default.",
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    from umich_slm.utils import set_seed
    set_seed(args.seed, deterministic=args.deterministic)
    print(f"Random seed: {args.seed}" + (" (deterministic mode)" if args.deterministic else ""))

    # Generate run name
    run_name = args.run_name or get_run_name(args)
    print(f"Run name: {run_name}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model configuration
    print("\n=== Creating Model Configuration ===")
    config_dict = create_model_config(args)
    print(f"Model config: {config_dict}")

    # Save initialized model
    model_path = output_dir / "models" / run_name
    if not args.config_only:
        print(f"\n=== Saving Model to {model_path} ===")
        save_model_init(config_dict, model_path, tokenizer_path=args.tokenizer_path)
    else:
        print(f"\n=== Skipping model initialization (--config_only) ===")
        # Just save the config
        model_path.mkdir(parents=True, exist_ok=True)
        from umich_slm import UMichSLMConfig
        config = UMichSLMConfig(**config_dict)
        config.save_pretrained(model_path)

    # Generate LLaMA Factory config
    print(f"\n=== Generating LLaMA Factory Config ===")
    lf_config = generate_llamafactory_config(args, model_path, run_name)

    config_path = output_dir / "configs" / f"{run_name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(lf_config, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to {config_path}")

    # Print dataset registration info
    print("\n=== Dataset Registration ===")
    print("To use SmolLM-corpus, add the following to your LLaMA Factory dataset_info.json:")
    dataset_info = register_smollm_dataset()
    print(yaml.dump(dataset_info, default_flow_style=False))

    # Run training
    if not args.dry_run and not args.config_only:
        print(f"\n=== Starting Training ===")
        return_code = run_training(config_path, dry_run=False)
        sys.exit(return_code)
    else:
        print(f"\n=== Dry Run Complete ===")
        print(f"To run training, execute:")
        print(f"  python scripts/llamafactory_train.py train {config_path}")
        print(f"  # Or ensure umich_slm is installed and use:")
        print(f"  # llamafactory-cli train {config_path}")


if __name__ == "__main__":
    main()
