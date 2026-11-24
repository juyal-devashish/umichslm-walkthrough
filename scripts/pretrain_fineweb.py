#!/usr/bin/env python3
"""Pretrain UMichSLM models with LLaMA Factory.

This script supports token-budgeted pretraining on any dataset that LLaMA Factory
can resolve via `--dataset` (and optional `--dataset_dir`), not just FineWeb.

The script:
1. Creates a UMichSLM model with the specified configuration.
2. Saves model/config/tokenizer artifacts for LLaMA Factory to load.
3. Generates a LLaMA Factory YAML config targeting a token budget.
4. Runs pretraining via the LLaMA Factory CLI.

Token budget calculation:
    - Default sequence length: 2048 tokens
    - Default target: 3 billion tokens (3B)
    - Default validation: 3% of total tokens (held out from training)
    - With batch_size=8, grad_accum=16: effective batch = 128

Reproducibility:
    By default, streaming is disabled and max_samples is set to ensure the same
    subset of data (first ~488K samples) is used across runs. Combined with a
    fixed seed (default: 42), this ensures reproducible data selection.

Example usage:
    # Basic pretraining with a preset dataset
    python scripts/pretrain_fineweb.py --size small --output_dir ./outputs/fineweb-1B

    # Using a local/custom dataset registration via --dataset and --dataset_dir
    python scripts/pretrain_fineweb.py --size small \\
        --dataset my_train_dataset \\
        --dataset_dir ./data/my_dataset \\
        --output_dir ./outputs/my-dataset-run

    # With validation during training
    python scripts/pretrain_fineweb.py --size small \\
        --dataset my_train_dataset \\
        --dataset_dir ./data/my_dataset \\
        --do_eval --eval_dataset my_val_dataset --eval_steps 500

    # With custom token target (in billions)
    python scripts/pretrain_fineweb.py --size small --target_tokens_b 2.5

    # With custom validation percentage
    python scripts/pretrain_fineweb.py --size small --target_tokens_b 3.0 --val_percent 5.0

    # With MoD enabled
    python scripts/pretrain_fineweb.py --size small --use_mod

    # Multi-GPU (LLaMA Factory handles this automatically via torchrun)
    torchrun --nproc_per_node=4 scripts/pretrain_fineweb.py --size base --num_gpus 4

    # Optional: with DeepSpeed ZeRO for additional memory optimization
    python scripts/pretrain_fineweb.py --size base --deepspeed configs/llama_factory/ds_config_zero2.json

    # Dry run to see config without training
    python scripts/pretrain_fineweb.py --size small --dry_run

    # More verbose logging (log every step)
    python scripts/pretrain_fineweb.py --size small --logging_steps 1 --logging_first_step

    # Enhanced logging with NaN/Inf filtering
    python scripts/pretrain_fineweb.py --size small --logging_steps 10 --logging_nan_inf_filter

    # Record loss for every step in WandB
    python scripts/pretrain_fineweb.py --size small --wandb_project my-project --wandb_log_every_step
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


def get_vocab_size_from_tokenizer(tokenizer_path: str) -> int:
    """Get the vocabulary size from a tokenizer.

    Args:
        tokenizer_path: Path to the tokenizer.

    Returns:
        Vocabulary size of the tokenizer, including added tokens.
        Uses len(tokenizer) to get the actual size, which includes
        special tokens that may be beyond the base vocab_size.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    # len(tokenizer) gives the actual vocabulary size including added tokens
    # This is important for tokenizers where special tokens may be added
    # beyond the base vocab_size
    return len(tokenizer)


def calculate_training_steps(
    target_tokens_b: float,
    val_percent: float,
    cutoff_len: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: int = 1,
) -> tuple[int, int, int, int]:
    """Calculate training steps and samples needed for target token count.

    Args:
        target_tokens_b: Target number of tokens in billions
        val_percent: Percentage of tokens for validation (held out from training)
        cutoff_len: Maximum sequence length (tokens per sample)
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        num_gpus: Number of GPUs (for distributed training)

    Returns:
        Tuple of (train_tokens, val_tokens, max_steps, max_samples)
        where train_tokens and val_tokens are in absolute numbers (not billions)
    """
    # Total tokens in absolute number
    total_tokens = int(target_tokens_b * 1_000_000_000)

    # Calculate validation tokens (held out from training)
    val_tokens = int(total_tokens * (val_percent / 100.0))
    train_tokens = total_tokens - val_tokens

    # Effective batch size in samples
    effective_batch_size = batch_size * gradient_accumulation_steps * num_gpus

    # Tokens per step
    tokens_per_step = effective_batch_size * cutoff_len

    # Calculate steps needed for training tokens
    max_steps = train_tokens // tokens_per_step

    # Calculate samples needed (with some buffer for variable-length sequences)
    max_samples = (train_tokens // cutoff_len) + effective_batch_size

    return train_tokens, val_tokens, max_steps, max_samples


def get_run_name(args: argparse.Namespace) -> str:
    """Generate a descriptive run name including the date."""
    # For custom architectures, put depth and width first for visibility
    if args.size == "custom" and args.num_hidden_layers is not None:
        # Format: deep{layers}L_{hidden}h for custom deep architectures
        parts = [f"deep{args.num_hidden_layers}L", f"{args.hidden_size}h"]
    else:
        parts = [args.size]

    # Token count (use M for < 1B so small runs don't show as "0.0B")
    tokens_m = args.target_tokens_b * 1000
    if tokens_m < 1000:
        parts.append(f"{tokens_m:.0f}Mtok")
    else:
        parts.append(f"{args.target_tokens_b:.1f}Btok")

    # Add validation percentage if not default
    if args.val_percent != 3.0:
        parts.append(f"val{args.val_percent:.0f}pct")

    # Position embedding type
    if args.position_embedding_type != "rope":
        parts.append(args.position_embedding_type)

    # Attention type (GQA if num_key_value_heads is set and different from default)
    if args.num_key_value_heads is not None and args.num_key_value_heads != args.num_attention_heads:
        parts.append(f"gqa{args.num_key_value_heads}")

    # Features
    if args.use_mod:
        parts.append(f"mod{int(args.mod_capacity * 100)}")
    if args.use_mtp:
        parts.append(f"mtp{args.mtp_num_tokens}")
    if args.use_moe:
        moe_str = f"moe{args.num_experts}x{args.num_experts_per_tok}"
        if args.moe_expert_intermediate_size is not None:
            moe_str += f"_exp{args.moe_expert_intermediate_size}"
        if args.moe_progressive:
            moe_str += "_prog"
        parts.append(moe_str)
    if args.use_mla:
        parts.append(f"mla{args.mla_kv_lora_rank}")
    if args.sliding_window:
        parts.append(f"sw{args.sliding_window}")
    if args.use_deltanet:
        deltanet_str = f"deltanet_{args.deltanet_variant}"
        if args.deltanet_every_n_layers > 0:
            deltanet_str += f"_every{args.deltanet_every_n_layers}"
        parts.append(deltanet_str)
    if args.use_zero_centered_rmsnorm:
        parts.append("zcrmsnorm")
    if args.use_repo:
        repo_str = f"repo{args.repo_start_layer}"
        parts.append(repo_str)

    # Add date
    date_str = datetime.now().strftime("%Y%m%d-%H%M")
    parts.append(date_str)

    return "_".join(parts)


def get_wandb_tags(args: argparse.Namespace) -> list[str]:
    """Generate WandB tags based on configuration for easy filtering."""
    tags = []

    # User who launched the job
    import getpass
    try:
        username = getpass.getuser()
        tags.append(f"user:{username}")
    except Exception:
        pass  # Skip if unable to get username

    # Model size
    tags.append(f"size:{args.size}")

    # Dataset token budget
    tokens_m = args.target_tokens_b * 1000
    if tokens_m < 1000:
        tags.append(f"tokens:{tokens_m:.0f}M")
    else:
        tags.append(f"tokens:{args.target_tokens_b:.1f}B")

    # Attention mechanism
    if args.use_mla:
        tags.append("attention:mla")
    elif args.use_deltanet:
        tags.append(f"attention:deltanet-{args.deltanet_variant}")
    elif args.num_key_value_heads is not None:
        tags.append("attention:gqa")
    else:
        tags.append("attention:mha")

    # Position embeddings
    tags.append(f"pos_emb:{args.position_embedding_type}")

    # Normalization
    if args.use_zero_centered_rmsnorm:
        tags.append("norm:zc-rmsnorm")
    else:
        tags.append("norm:rmsnorm")

    # MLP type
    if args.use_moe:
        if args.moe_progressive:
            tags.append("mlp:progressive-moe")
        else:
            tags.append("mlp:moe")
        tags.append(f"moe:experts-{args.num_experts}")
        tags.append(f"moe:topk-{args.num_experts_per_tok}")
        if args.moe_expert_intermediate_size:
            tags.append("moe:param-matched")
    else:
        tags.append("mlp:swiglu")

    # Compute efficiency features
    if args.use_mod:
        tags.append("efficiency:mod")
        tags.append(f"mod:capacity-{int(args.mod_capacity * 100)}pct")
    if args.use_mtp:
        tags.append("efficiency:mtp")
        tags.append(f"mtp:heads-{args.mtp_num_tokens}")
    if args.sliding_window:
        tags.append("efficiency:sliding-window")
        tags.append(f"swa:window-{args.sliding_window}")

    # DeltaNet specific tags
    if args.use_deltanet:
        tags.append(f"deltanet:variant-{args.deltanet_variant}")
        if args.deltanet_every_n_layers > 0:
            tags.append(f"deltanet:hybrid-every-{args.deltanet_every_n_layers}")
        if args.deltanet_first_n_layers > 0:
            tags.append(f"deltanet:first-{args.deltanet_first_n_layers}-std")

    # RePo (learned position embeddings)
    if args.use_repo:
        tags.append("position:repo")
        tags.append(f"repo:start-layer-{args.repo_start_layer}")
        if args.repo_use_base_positions:
            tags.append("repo:with-base")
        else:
            tags.append("repo:pure-learned")

    # Training config
    tags.append(f"lr:{args.learning_rate:.0e}")
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    tags.append(f"batch:{effective_batch}")

    return tags


def get_wandb_group(args: argparse.Namespace) -> str:
    """Generate WandB group name for grouping related experiments."""
    # Group by main architectural choices
    parts = []

    # Attention type
    if args.use_mla:
        parts.append("mla")
    elif args.use_deltanet:
        parts.append(f"deltanet-{args.deltanet_variant}")
    elif args.num_key_value_heads is not None:
        parts.append("gqa")
    else:
        parts.append("mha")

    # MLP type
    if args.use_moe:
        if args.moe_progressive:
            parts.append("prog-moe")
        else:
            parts.append("moe")
    else:
        parts.append("dense")

    # Efficiency features
    efficiency = []
    if args.use_mod:
        efficiency.append("mod")
    if args.use_mtp:
        efficiency.append("mtp")
    if args.sliding_window:
        efficiency.append("swa")
    if efficiency:
        parts.append("+".join(efficiency))

    # Special features
    if args.position_embedding_type == "pope":
        parts.append("pope")
    if args.use_repo:
        parts.append("repo")
    if args.use_zero_centered_rmsnorm:
        parts.append("zc-norm")

    return "_".join(parts) if parts else "baseline"


def create_model_config(args: argparse.Namespace, vocab_size: Optional[int] = None) -> dict:
    """Create model configuration dictionary.

    Args:
        args: Command line arguments.
        vocab_size: Vocabulary size from the tokenizer. REQUIRED - must be provided
            to ensure the model matches the tokenizer being used. The preset's
            hardcoded vocab_size is ignored and replaced with this value.
    """
    from umich_slm.configs.presets import UMICHSLM_CONFIGS

    # Handle custom model size for scaling experiments
    if args.size == "custom":
        # Validate required custom args
        required_custom = ["hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"]
        missing = [arg for arg in required_custom if getattr(args, arg) is None]
        if missing:
            raise ValueError(
                f"When using --size=custom, the following arguments are required: {missing}"
            )

        # Build custom config from scratch
        config = {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "num_attention_heads": args.num_attention_heads,
            "intermediate_size": args.intermediate_size,
            "num_key_value_heads": args.num_key_value_heads or args.num_attention_heads,  # Default to MHA
            "hidden_act": "silu",
            "max_position_embeddings": 2048,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "use_cache": True,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": False,
            "rope_theta": 10000.0,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "mlp_bias": False,
        }
    else:
        # Start with preset config
        if args.size not in UMICHSLM_CONFIGS:
            raise ValueError(f"Unknown size: {args.size}. Available: {list(UMICHSLM_CONFIGS.keys())}")

        config = UMICHSLM_CONFIGS[args.size].copy()

    # vocab_size MUST come from the tokenizer, not from the preset
    # The preset's vocab_size is just a placeholder and should always be overridden
    if vocab_size is None:
        raise ValueError(
            "vocab_size must be provided (from tokenizer). "
            "The preset's hardcoded vocab_size should not be used. "
            "Use get_vocab_size_from_tokenizer() to get the correct value."
        )
    config["vocab_size"] = vocab_size

    # Position embedding
    config["position_embedding_type"] = args.position_embedding_type
    config["rope_theta"] = args.rope_theta

    # Helper to check if a flag was explicitly passed on the CLI.
    # For store_true flags, argparse sets them to False by default, which
    # would silently override preset values that are True.  We only want
    # to override when the user *explicitly* passed the flag.
    import sys as _sys
    _explicit = set()
    for token in _sys.argv[1:]:
        _explicit.add(token.lstrip("-").replace("-", "_"))

    def _was_set(name: str) -> bool:
        return name in _explicit

    # Mixture of Depths — only override if explicitly passed
    if _was_set("use_mod"):
        config["use_mod"] = args.use_mod
    if args.use_mod or config.get("use_mod", False):
        if _was_set("mod_capacity"):
            config["mod_capacity"] = args.mod_capacity
        if _was_set("mod_every_n_layers"):
            config["mod_every_n_layers"] = args.mod_every_n_layers
        if _was_set("mod_aux_loss_weight"):
            config["mod_aux_loss_weight"] = args.mod_aux_loss_weight

    # Multi-Token Prediction
    if _was_set("use_mtp"):
        config["use_mtp"] = args.use_mtp
    if args.use_mtp or config.get("use_mtp", False):
        if _was_set("mtp_num_tokens"):
            config["mtp_num_heads"] = args.mtp_num_tokens
        if _was_set("mtp_loss_weight"):
            config["mtp_loss_weight"] = args.mtp_loss_weight

    # Token-Superposition Training (TST). The init config records the flags;
    # llamafactory_train.py reads them off model.config and attaches the
    # TSTTrainerCallback at HF Trainer instantiation time.
    if _was_set("tst_enabled"):
        config["tst_enabled"] = bool(args.tst_enabled)
    if args.tst_enabled or config.get("tst_enabled", False):
        if _was_set("tst_bag_size"):
            config["tst_bag_size"] = int(args.tst_bag_size)
        if _was_set("tst_recovery_ratio"):
            config["tst_recovery_ratio"] = float(args.tst_recovery_ratio)
        if _was_set("tst_loss_weighting"):
            config["tst_loss_weighting"] = args.tst_loss_weighting

    # Delphi recipe: stash run-context fields onto the model config so the
    # Trainer-init patch in llamafactory_train.py can swap optimizer/scheduler.
    if args.delphi_recipe:
        config["delphi_recipe"] = True
        config["delphi_optimizer"] = args.delphi_optimizer
        config["delphi_schedule"] = args.delphi_schedule
        config["delphi_warmup_frac"] = float(args.delphi_warmup_frac)
        config["delphi_decay_frac"] = float(args.delphi_decay_frac)
        config["delphi_eta0_proj"] = float(args.delphi_eta0_proj)
        config["delphi_eta0_adam"] = float(args.delphi_eta0_adam)
        config["delphi_batch_ref"] = int(args.delphi_batch_ref)
        config["delphi_token_ref"] = float(args.delphi_token_ref)
        config["delphi_horizon_exp"] = float(args.delphi_horizon_exp)
        config["delphi_epsilon0"] = float(args.delphi_epsilon0)
        config["delphi_beta2_max"] = float(args.delphi_beta2_max)
        config["delphi_beta2_min"] = float(args.delphi_beta2_min)
        config["delphi_grad_clip"] = float(args.delphi_grad_clip)
        # Run-context (filled in deeper in main() after max_steps is known —
        # see the "delphi_run_*" block).

    # Mixture of Experts
    if _was_set("use_moe"):
        config["use_moe"] = args.use_moe
    if args.use_moe or config.get("use_moe", False):
        if _was_set("num_experts"):
            config["num_experts"] = args.num_experts
        if _was_set("num_experts_per_tok"):
            config["num_experts_per_tok"] = args.num_experts_per_tok
        if _was_set("num_shared_experts"):
            config["num_shared_experts"] = args.num_shared_experts
        if _was_set("moe_expert_intermediate_size"):
            config["moe_expert_intermediate_size"] = args.moe_expert_intermediate_size
        if _was_set("moe_progressive"):
            config["moe_progressive"] = args.moe_progressive
        if _was_set("moe_progressive_start_ratio"):
            config["moe_progressive_start_ratio"] = args.moe_progressive_start_ratio

    # Sliding Window Attention
    if _was_set("sliding_window"):
        config["sliding_window"] = args.sliding_window
    if args.sliding_window or config.get("sliding_window"):
        if _was_set("global_attention_every_n_layers"):
            config["global_attention_every_n_layers"] = args.global_attention_every_n_layers

    # DeltaNet (Linear Attention)
    if _was_set("use_deltanet"):
        config["use_deltanet"] = args.use_deltanet
    if args.use_deltanet or config.get("use_deltanet", False):
        if _was_set("deltanet_variant"):
            config["deltanet_variant"] = args.deltanet_variant
        if _was_set("deltanet_every_n_layers"):
            config["deltanet_every_n_layers"] = args.deltanet_every_n_layers
        if _was_set("deltanet_first_n_layers"):
            config["deltanet_first_n_layers"] = args.deltanet_first_n_layers
        if _was_set("deltanet_expand_k"):
            config["deltanet_expand_k"] = args.deltanet_expand_k
        if _was_set("deltanet_expand_v"):
            config["deltanet_expand_v"] = args.deltanet_expand_v

    # Multi-head Latent Attention (MLA)
    if _was_set("use_mla"):
        config["use_mla"] = args.use_mla
    if args.use_mla or config.get("use_mla", False):
        if _was_set("mla_kv_lora_rank"):
            config["mla_kv_lora_rank"] = args.mla_kv_lora_rank
        if _was_set("mla_q_lora_rank"):
            config["mla_q_lora_rank"] = args.mla_q_lora_rank
        if _was_set("mla_qk_nope_head_dim"):
            config["mla_qk_nope_head_dim"] = args.mla_qk_nope_head_dim
        if _was_set("mla_qk_rope_head_dim"):
            config["mla_qk_rope_head_dim"] = args.mla_qk_rope_head_dim
        if _was_set("mla_v_head_dim"):
            config["mla_v_head_dim"] = args.mla_v_head_dim

    # RePo (Re-Positioning) - Learned position embeddings
    if _was_set("use_repo"):
        config["use_repo"] = args.use_repo
    if args.use_repo or config.get("use_repo", False):
        if _was_set("repo_dim"):
            config["repo_dim"] = args.repo_dim
        if _was_set("repo_start_layer"):
            config["repo_start_layer"] = args.repo_start_layer
        if _was_set("repo_use_base_positions") or _was_set("repo_no_base_positions"):
            config["repo_use_base_positions"] = args.repo_use_base_positions

    # Training settings
    config["attention_dropout"] = args.attention_dropout
    if _was_set("tie_word_embeddings"):
        config["tie_word_embeddings"] = args.tie_word_embeddings

    # Attention type (MHA vs GQA)
    if args.num_key_value_heads is not None:
        config["num_key_value_heads"] = args.num_key_value_heads

    # Zero-centered RMSNorm
    if _was_set("use_zero_centered_rmsnorm"):
        config["use_zero_centered_rmsnorm"] = args.use_zero_centered_rmsnorm

    return config


def save_model_init(config_dict: dict, output_path: Path, tokenizer_path: Optional[str] = None, dtype: Optional[str] = None) -> None:
    """Save initialized model for LLaMA Factory to load.

    Args:
        config_dict: Model configuration dictionary
        output_path: Path to save the model
        tokenizer_path: Optional path to tokenizer
        dtype: Optional dtype to save model in ('bf16', 'fp16', or None for fp32)
    """
    import torch
    from umich_slm import UMichSLMConfig, UMichSLMForCausalLM
    from transformers import AutoTokenizer

    output_path.mkdir(parents=True, exist_ok=True)

    # Load tokenizer first to get the correct vocab_size
    if tokenizer_path:
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    else:
        # Default to the project SuperBPE-56000 tokenizer (see
        # umich_slm.tokenizer_defaults / TOKENIZER_TRAINING_PLAN.md).
        from umich_slm.tokenizer_defaults import get_tokenizer_path
        default_tokenizer_path = get_tokenizer_path()
        print(f"Loading tokenizer from {default_tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(default_tokenizer_path, trust_remote_code=True)

    # Get actual vocabulary size including added tokens
    # len(tokenizer) gives the total size, which includes special tokens
    # that may be beyond the base vocab_size
    actual_vocab_size = len(tokenizer)
    base_vocab_size = tokenizer.vocab_size

    if actual_vocab_size != base_vocab_size:
        print(f"Tokenizer has {actual_vocab_size - base_vocab_size} added tokens beyond base vocab_size {base_vocab_size}")
        print(f"Using actual vocab_size {actual_vocab_size} (includes added tokens)")

    # Update config_dict to match actual tokenizer vocab_size (including added tokens)
    # ALWAYS update to match tokenizer, even if they appear to match
    # This ensures consistency
    old_vocab_size = config_dict.get("vocab_size", 32000)
    if actual_vocab_size != old_vocab_size:
        print(f"Updating model vocab_size from {old_vocab_size} to {actual_vocab_size} to match tokenizer")
        config_dict["vocab_size"] = actual_vocab_size
    else:
        print(f"Model vocab_size ({actual_vocab_size}) matches tokenizer vocab_size")

    # Update special token IDs to match tokenizer
    # This prevents warnings from LLaMA Factory about mismatched special tokens
    tokenizer_bos_id = tokenizer.bos_token_id
    tokenizer_eos_id = tokenizer.eos_token_id
    tokenizer_pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Validate that special token IDs are within valid vocabulary range
    # Use actual_vocab_size which includes added tokens
    if tokenizer_bos_id is not None and (tokenizer_bos_id < 0 or tokenizer_bos_id >= actual_vocab_size):
        raise ValueError(
            f"Tokenizer bos_token_id ({tokenizer_bos_id}) is out of range [0, {actual_vocab_size})"
        )
    if tokenizer_eos_id is not None and (tokenizer_eos_id < 0 or tokenizer_eos_id >= actual_vocab_size):
        raise ValueError(
            f"Tokenizer eos_token_id ({tokenizer_eos_id}) is out of range [0, {actual_vocab_size})"
        )
    if tokenizer_pad_id is not None and (tokenizer_pad_id < 0 or tokenizer_pad_id >= actual_vocab_size):
        raise ValueError(
            f"Tokenizer pad_token_id ({tokenizer_pad_id}) is out of range [0, {actual_vocab_size})"
        )

    old_bos_id = config_dict.get("bos_token_id", 1)
    old_eos_id = config_dict.get("eos_token_id", 2)
    old_pad_id = config_dict.get("pad_token_id", 0)

    if tokenizer_bos_id != old_bos_id:
        print(f"Updating bos_token_id from {old_bos_id} to {tokenizer_bos_id} to match tokenizer")
        config_dict["bos_token_id"] = tokenizer_bos_id
    if tokenizer_eos_id != old_eos_id:
        print(f"Updating eos_token_id from {old_eos_id} to {tokenizer_eos_id} to match tokenizer")
        config_dict["eos_token_id"] = tokenizer_eos_id
    if tokenizer_pad_id != old_pad_id:
        print(f"Updating pad_token_id from {old_pad_id} to {tokenizer_pad_id} to match tokenizer")
        config_dict["pad_token_id"] = tokenizer_pad_id

    # Create config and model with correct vocab_size and special token IDs
    print(f"Creating model config with vocab_size={config_dict.get('vocab_size')} from config_dict")
    config = UMichSLMConfig(**config_dict)
    print(f"Config created with vocab_size={config.vocab_size}")

    # Double-check that config has the correct vocab_size and special token IDs
    if config.vocab_size != actual_vocab_size:
        print(f"WARNING: Config vocab_size ({config.vocab_size}) != tokenizer vocab_size ({actual_vocab_size})")
        print(f"Force updating config vocab_size to {actual_vocab_size}")
        config.vocab_size = actual_vocab_size
        print(f"Config vocab_size updated to {config.vocab_size}")

    # Verify special token IDs match
    if config.bos_token_id != tokenizer_bos_id:
        print(f"WARNING: Config bos_token_id ({config.bos_token_id}) != tokenizer bos_token_id ({tokenizer_bos_id})")
        print(f"Force updating config bos_token_id to {tokenizer_bos_id}")
        config.bos_token_id = tokenizer_bos_id
    if config.eos_token_id != tokenizer_eos_id:
        print(f"WARNING: Config eos_token_id ({config.eos_token_id}) != tokenizer eos_token_id ({tokenizer_eos_id})")
        print(f"Force updating config eos_token_id to {tokenizer_eos_id}")
        config.eos_token_id = tokenizer_eos_id
    if config.pad_token_id != tokenizer_pad_id:
        print(f"WARNING: Config pad_token_id ({config.pad_token_id}) != tokenizer pad_token_id ({tokenizer_pad_id})")
        print(f"Force updating config pad_token_id to {tokenizer_pad_id}")
        config.pad_token_id = tokenizer_pad_id

    print(f"Creating model with config vocab_size={config.vocab_size}")
    model = UMichSLMForCausalLM(config)
    print(f"Model created with vocab_size={model.vocab_size}, embedding size={model.get_input_embeddings().num_embeddings}")

    # Verify model was created with correct vocab_size
    if model.vocab_size != actual_vocab_size:
        raise ValueError(
            f"Model vocab_size ({model.vocab_size}) != tokenizer vocab_size ({actual_vocab_size}). "
            f"This will cause embedding lookup errors. Please check the model initialization."
        )

    # Verify embedding layers match
    input_emb_size = model.get_input_embeddings().num_embeddings
    # Output embeddings are a Linear layer, so check out_features instead of num_embeddings
    output_emb_size = model.get_output_embeddings().out_features
    if input_emb_size != actual_vocab_size:
        raise ValueError(
            f"Input embedding size ({input_emb_size}) != tokenizer vocab_size ({actual_vocab_size})"
        )
    if output_emb_size != actual_vocab_size:
        raise ValueError(
            f"Output embedding size ({output_emb_size}) != tokenizer vocab_size ({actual_vocab_size})"
        )
    print(f"✓ Model vocab_size and embedding sizes match tokenizer ({actual_vocab_size})")

    # Align tokenizer context limit with the model (avoids HF warnings and bad checks
    # when raw text > SmolLM2's default 8192 but model max_position_embeddings differs).
    mpe = int(getattr(config, "max_position_embeddings", 2048))
    old_mml = getattr(tokenizer, "model_max_length", None)
    if old_mml is not None and old_mml != mpe:
        print(f"Setting tokenizer.model_max_length {old_mml} -> {mpe} (model max_position_embeddings)")
    tokenizer.model_max_length = mpe

    # Convert model to specified dtype before saving
    if dtype:
        if dtype == "bf16":
            torch_dtype = torch.bfloat16
            dtype_str = "bfloat16"
        elif dtype == "fp16":
            torch_dtype = torch.float16
            dtype_str = "float16"
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Use 'bf16' or 'fp16'")

        print(f"Converting model to {dtype_str} before saving...")
        model = model.to(torch_dtype)
    else:
        dtype_str = "float32"
        print(f"Model will be saved in {dtype_str} (no dtype conversion)")

    # Save model and config
    # The dtype is preserved in the state dict when the model is converted before saving
    print(f"Saving model with vocab_size={model.vocab_size} to {output_path}")
    model.save_pretrained(output_path)
    print(f"Model saved to {output_path} in {dtype_str}")

    # Verify saved config has correct vocab_size
    import json
    saved_config_path = output_path / "config.json"
    if saved_config_path.exists():
        with open(saved_config_path, "r") as f:
            saved_config = json.load(f)
        saved_vocab_size = saved_config.get("vocab_size")
        print(f"Saved config.json has vocab_size={saved_vocab_size}")
        if saved_vocab_size != actual_vocab_size:
            print(f"⚠️  WARNING: Saved config vocab_size ({saved_vocab_size}) != expected ({actual_vocab_size})")
        else:
            print(f"✓ Saved config vocab_size matches expected value")

    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    print(f"Tokenizer saved to {output_path}")

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # CRITICAL: Explicitly delete model and clear CUDA cache to free GPU memory
    # Without this, the model stays in memory when LLaMA Factory loads its own copy,
    # causing OOM even for small models
    del model
    del config
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared after model save")


def generate_llamafactory_config(
    args: argparse.Namespace,
    model_path: Path,
    run_name: str,
    max_steps: int,
    max_samples: int,
) -> dict:
    """Generate a LLaMA Factory YAML configuration for pretraining."""
    # Determine dataset directory
    if args.dataset_dir:
        dataset_dir = str(Path(args.dataset_dir).absolute())
    else:
        dataset_dir = str(Path(__file__).parent.parent.absolute() / "configs" / "llama_factory")

    config = {
        # Model Configuration
        "model_name_or_path": str(model_path),
        "trust_remote_code": True,

        # Data Configuration
        "dataset": args.dataset,
        "cutoff_len": args.cutoff_len,
        "preprocessing_num_workers": args.num_workers,
        "streaming": args.streaming,
        "dataset_dir": dataset_dir,

        # Training Stage
        "stage": "pt",  # Pretraining
        "do_train": True,
        "finetuning_type": "full",

        # Output Configuration
        "output_dir": str(Path(args.output_dir) / run_name),
        "logging_dir": str(Path(args.output_dir) / run_name / "logs"),
        "overwrite_output_dir": True,
        "run_name": run_name,

        # Training Hyperparameters - targeting specific token count
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_steps": max_steps,
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
        "save_total_limit": args.save_total_limit,

        # Misc
        "seed": args.seed,
        "dataloader_num_workers": args.num_workers,
        "dataloader_pin_memory": True,
    }

    # For non-streaming mode, limit samples
    if not args.streaming:
        config["max_samples"] = max_samples

    # Wandb integration
    # Note: wandb_project and wandb_entity are set via environment variables
    # rather than config keys, as LLaMA Factory doesn't accept them directly
    if args.wandb_project:
        config["report_to"] = "wandb"
    else:
        config["report_to"] = "tensorboard"

    # DeepSpeed for distributed training
    if args.deepspeed:
        config["deepspeed"] = args.deepspeed

    # Evaluation configuration
    if args.do_eval and args.eval_dataset:
        config["do_eval"] = True
        config["eval_dataset"] = args.eval_dataset
        config["eval_steps"] = args.eval_steps
        config["eval_strategy"] = "steps"
        # Use same dataset_dir for eval dataset
        # (assumes train and val are in same directory)

    return config


def run_training(
    config_path: Path,
    dry_run: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_tags: Optional[list[str]] = None,
    wandb_group: Optional[str] = None,
) -> int:
    """Run LLaMA Factory training.

    Args:
        config_path: Path to the YAML config file
        dry_run: If True, only print the command without running
        wandb_project: WandB project name (set via environment variable)
        wandb_entity: WandB entity name (set via environment variable)
        wandb_tags: List of tags for WandB run (set via environment variable)
        wandb_group: Group name for WandB run (set via environment variable)
    """
    script_dir = Path(__file__).parent
    wrapper_script = script_dir / "llamafactory_train.py"

    if wrapper_script.exists():
        cmd = [sys.executable, str(wrapper_script), "train", str(config_path)]
    else:
        print("Warning: Wrapper script not found, using llamafactory-cli directly")
        cmd = ["llamafactory-cli", "train", str(config_path)]

    if dry_run:
        env_info = []
        if wandb_project:
            env_info.append(f"WANDB_PROJECT={wandb_project}")
        if wandb_entity:
            env_info.append(f"WANDB_ENTITY={wandb_entity}")
        if wandb_tags:
            env_info.append(f"WANDB_TAGS={','.join(wandb_tags)}")
        if wandb_group:
            env_info.append(f"WANDB_RUN_GROUP={wandb_group}")
        env_str = " ".join(env_info) + " " if env_info else ""
        print(f"Dry run - would execute: {env_str}{' '.join(cmd)}")
        return 0

    # Set WandB environment variables if provided
    env = os.environ.copy()
    if wandb_project:
        env["WANDB_PROJECT"] = wandb_project
        print(f"Setting WANDB_PROJECT={wandb_project}")
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
        print(f"Setting WANDB_ENTITY={wandb_entity}")
    if wandb_tags:
        # WandB expects tags as comma-separated string
        env["WANDB_TAGS"] = ",".join(wandb_tags)
        print(f"Setting WANDB_TAGS={env['WANDB_TAGS']}")
    if wandb_group:
        env["WANDB_RUN_GROUP"] = wandb_group
        print(f"Setting WANDB_RUN_GROUP={wandb_group}")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main():
    from umich_slm.configs.presets import UMICHSLM_CONFIGS

    parser = argparse.ArgumentParser(
        description="Pretrain UMichSLM on a token budget using LLaMA Factory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Token target
    parser.add_argument(
        "--target_tokens_b",
        type=float,
        default=3.0,
        help="Target number of tokens in billions (default: 3.0, i.e., 3 billion tokens)",
    )
    parser.add_argument(
        "--val_percent",
        type=float,
        default=3.0,
        help="Percentage of tokens to use for validation, held out from training (default: 3.0%%)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs (only used to calculate max_steps correctly; actual multi-GPU is handled automatically by LLaMA Factory when launched with torchrun)",
    )

    # Dataset configuration
    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument(
        "--dataset",
        type=str,
        default="fineweb_edu",
        help="Training dataset name recognized by LLaMA Factory",
    )
    data_group.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Optional dataset directory containing dataset metadata/configs",
    )
    data_group.add_argument(
        "--eval_dataset",
        type=str,
        default=None,
        help="Optional evaluation dataset name",
    )
    data_group.add_argument(
        "--do_eval",
        action="store_true",
        help="Run evaluation during training",
    )
    data_group.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluation interval (in steps)",
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--size",
        type=str,
        default="small",
        choices=[*UMICHSLM_CONFIGS.keys(), "custom"],
        help="Model size preset. Use 'custom' with --hidden_size etc. for scaling experiments.",
    )
    # Custom model size arguments (for scaling law experiments)
    model_group.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="Hidden size (required if --size=custom)",
    )
    model_group.add_argument(
        "--num_hidden_layers",
        type=int,
        default=None,
        help="Number of layers (required if --size=custom)",
    )
    model_group.add_argument(
        "--num_attention_heads",
        type=int,
        default=None,
        help="Number of attention heads (required if --size=custom)",
    )
    model_group.add_argument(
        "--intermediate_size",
        type=int,
        default=None,
        help="MLP intermediate size (required if --size=custom)",
    )
    model_group.add_argument(
        "--position_embedding_type",
        type=str,
        default="rope",
        choices=["rope", "pope"],
        help="Position embedding type",
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
    model_group.add_argument(
        "--num_key_value_heads",
        type=int,
        default=None,
        help="Number of key-value heads for grouped-query attention (GQA). If None, uses preset default. Set to num_attention_heads for MHA.",
    )
    model_group.add_argument(
        "--use_zero_centered_rmsnorm",
        action="store_true",
        help="Use zero-centered RMSNorm (subtracts mean before normalization, as in Gemma 2)",
    )

    # Mixture of Depths
    mod_group = parser.add_argument_group("Mixture of Depths")
    mod_group.add_argument("--use_mod", action="store_true", help="Enable Mixture of Depths")
    mod_group.add_argument("--mod_capacity", type=float, default=0.125, help="MoD capacity")
    mod_group.add_argument("--mod_every_n_layers", type=int, default=2, help="MoD frequency")
    mod_group.add_argument("--mod_aux_loss_weight", type=float, default=0.01, help="MoD aux loss weight")

    # Token-Superposition Training (Peng et al., arXiv:2605.06546)
    tst_group = parser.add_argument_group("Token-Superposition Training")
    tst_group.add_argument("--tst_enabled", action="store_true",
                           help="Enable TST phase (i) + recovery.")
    tst_group.add_argument("--tst_bag_size", type=int, default=6,
                           help="TST bag size s (paper: 4-8 dense, 16 for 10B MoE).")
    tst_group.add_argument("--tst_recovery_ratio", type=float, default=0.3,
                           help="Fraction of total steps in TST phase (i) before recovery.")
    tst_group.add_argument("--tst_loss_weighting", type=str, default="uniform",
                           choices=["uniform", "power_law"],
                           help="Within-bag MCE weighting (power_law for s>=8).")

    # Delphi-style scaling recipe (openathena.ai/blog/delphi)
    delphi_group = parser.add_argument_group("Delphi recipe (token-horizon LR, AdamH, WSD)")
    delphi_group.add_argument("--delphi_recipe", action="store_true",
                              help="Enable Delphi recipe: AdamH + WSD + token-horizon LR. "
                                   "Overrides --optimizer/--lr_scheduler/--learning_rate/--weight_decay/"
                                   "--warmup_ratio/--max_grad_norm.")
    delphi_group.add_argument("--delphi_optimizer", type=str, default="adamh", choices=["adamh", "adamw"])
    delphi_group.add_argument("--delphi_schedule", type=str, default="wsd",
                              choices=["wsd", "cosine", "linear"])
    delphi_group.add_argument("--delphi_warmup_frac", type=float, default=0.10)
    delphi_group.add_argument("--delphi_decay_frac", type=float, default=0.20)
    delphi_group.add_argument("--delphi_eta0_proj", type=float, default=6.30e-3)
    delphi_group.add_argument("--delphi_eta0_adam", type=float, default=6.56e-4)
    delphi_group.add_argument("--delphi_batch_ref", type=int, default=64)
    delphi_group.add_argument("--delphi_token_ref", type=float, default=2.5e9)
    delphi_group.add_argument("--delphi_horizon_exp", type=float, default=0.30)
    delphi_group.add_argument("--delphi_epsilon0", type=float, default=1.85e-8)
    delphi_group.add_argument("--delphi_beta2_max", type=float, default=0.9999)
    delphi_group.add_argument("--delphi_beta2_min", type=float, default=0.9)
    delphi_group.add_argument("--delphi_grad_clip", type=float, default=0.1)

    # Multi-Token Prediction
    mtp_group = parser.add_argument_group("Multi-Token Prediction")
    mtp_group.add_argument("--use_mtp", action="store_true", help="Enable MTP")
    mtp_group.add_argument("--mtp_num_tokens", type=int, default=2, help="Number of additional tokens to predict beyond the next token (e.g., 1 = predict 2 tokens total, 2 = predict 3 tokens total)")
    mtp_group.add_argument("--mtp_loss_weight", type=float, default=0.3, help="MTP loss weight")

    # Mixture of Experts
    moe_group = parser.add_argument_group("Mixture of Experts")
    moe_group.add_argument("--use_moe", action="store_true", help="Enable MoE")
    moe_group.add_argument("--num_experts", type=int, default=8, help="Number of experts (max for progressive MoE)")
    moe_group.add_argument("--num_experts_per_tok", type=int, default=2, help="Experts per token")
    moe_group.add_argument("--num_shared_experts", type=int, default=0, help="Shared experts")
    moe_group.add_argument("--moe_expert_intermediate_size", type=int, default=None,
                          help="Expert MLP intermediate size. None=use intermediate_size (8x params). "
                               "Set to intermediate_size//num_experts for parameter-matched MoE.")
    moe_group.add_argument("--moe_progressive", action="store_true",
                          help="Progressive MoE: first half uses plain MLP, then experts increase (2->4->8->...)")
    moe_group.add_argument("--moe_progressive_start_ratio", type=float, default=0.5,
                          help="Fraction of layers using plain MLP before MoE begins (default: 0.5 = first half)")

    # Sliding Window Attention
    swa_group = parser.add_argument_group("Sliding Window Attention")
    swa_group.add_argument("--sliding_window", type=int, default=None, help="Sliding window size")
    swa_group.add_argument("--global_attention_every_n_layers", type=int, default=4, help="Global attention frequency")

    # DeltaNet (Linear Attention)
    deltanet_group = parser.add_argument_group("DeltaNet (Linear Attention)")
    deltanet_group.add_argument("--use_deltanet", action="store_true", help="Enable DeltaNet (linear attention)")
    deltanet_group.add_argument("--deltanet_variant", type=str, default="gated", choices=["gated", "kda"], help="DeltaNet variant: 'gated' (Qwen3-Next) or 'kda' (Kimi-K2)")
    deltanet_group.add_argument("--deltanet_every_n_layers", type=int, default=0, help="Use DeltaNet every N layers (0 = all layers use DeltaNet)")
    deltanet_group.add_argument("--deltanet_first_n_layers", type=int, default=0, help="First N layers use standard attention (before DeltaNet layers)")
    deltanet_group.add_argument("--deltanet_expand_k", type=int, default=1, help="Key expansion factor for DeltaNet")
    deltanet_group.add_argument("--deltanet_expand_v", type=int, default=1, help="Value expansion factor for DeltaNet")

    # Multi-head Latent Attention (MLA)
    mla_group = parser.add_argument_group("Multi-head Latent Attention (MLA)")
    mla_group.add_argument("--use_mla", action="store_true", help="Enable Multi-head Latent Attention (DeepSeek V2/V3 style)")
    mla_group.add_argument("--mla_kv_lora_rank", type=int, default=512, help="Rank of the low-rank KV compression (controls KV cache size)")
    mla_group.add_argument("--mla_q_lora_rank", type=int, default=0, help="Rank of the low-rank Q compression (0 = no Q compression)")
    mla_group.add_argument("--mla_qk_nope_head_dim", type=int, default=128, help="Dimension of the non-position-encoded part of Q/K heads")
    mla_group.add_argument("--mla_qk_rope_head_dim", type=int, default=64, help="Dimension of the RoPE-encoded part of Q/K heads")
    mla_group.add_argument("--mla_v_head_dim", type=int, default=128, help="Dimension of value heads")

    # RePo (Re-Positioning) - Learned position embeddings
    repo_group = parser.add_argument_group("RePo (Re-Positioning)")
    repo_group.add_argument("--use_repo", action="store_true", help="Enable RePo learned position embeddings")
    repo_group.add_argument("--repo_dim", type=int, default=None, help="RePo representation dimension (default: hidden_size // 8)")
    repo_group.add_argument("--repo_start_layer", type=int, default=5, help="First layer to use RePo (earlier layers use standard RoPE)")
    repo_group.add_argument("--repo_use_base_positions", action="store_true", default=True, help="Add learned deltas to base sequential positions (more stable)")
    repo_group.add_argument("--repo_no_base_positions", action="store_false", dest="repo_use_base_positions", help="Use pure learned positions (no sequential base)")

    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    train_group.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    train_group.add_argument("--lr_scheduler", type=str, default="cosine", help="LR scheduler")
    train_group.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    train_group.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    train_group.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    train_group.add_argument("--optimizer", type=str, default="adamw_torch", help="Optimizer")
    train_group.add_argument("--cutoff_len", type=int, default=2048, help="Maximum sequence length")
    train_group.add_argument("--streaming", action="store_true", default=False, help="Use streaming mode (less reproducible, but lower memory)")
    train_group.add_argument("--no_streaming", action="store_false", dest="streaming", help="Disable streaming mode (default, more reproducible)")

    # Precision and optimization
    opt_group = parser.add_argument_group("Precision and Optimization")
    opt_group.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16")
    opt_group.add_argument("--fp16", action="store_true", help="Use float16")
    opt_group.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Gradient checkpointing")
    opt_group.add_argument("--flash_attn", action=argparse.BooleanOptionalAction, default=True,
                           help="Flash Attention 2 (use --no-flash_attn on nodes without flash-attn; falls back to sdpa)")
    opt_group.add_argument("--deepspeed", type=str, default=None, help="Optional DeepSpeed config path for ZeRO memory optimization (not required for multi-GPU training)")

    # Logging and saving
    log_group = parser.add_argument_group("Logging and Saving")
    log_group.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    log_group.add_argument("--logging_steps", type=int, default=50, help="Logging interval")
    log_group.add_argument("--save_steps", type=int, default=100, help="Checkpoint interval")
    log_group.add_argument("--save_total_limit", type=int, default=5, help="Max checkpoints to keep")

    # Wandb
    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument("--wandb_project", type=str, default="umich-slm", help="Wandb project")
    wandb_group.add_argument("--wandb_entity", type=str, default="umich-slm", help="Wandb entity")
    wandb_group.add_argument("--run_name", type=str, default=None, help="Custom run name")
    wandb_group.add_argument("--wandb_log_every_step", action="store_true", default=True, help="Record loss/metrics in WandB for every training step (sets logging_steps=1)")

    # Misc
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--seed", type=int, default=42, help="Random seed")
    misc_group.add_argument("--deterministic", action="store_true", help="Enable fully deterministic mode (may slow training by 5-15%%)")
    misc_group.add_argument("--num_workers", type=int, default=8, help="Data loading workers")
    misc_group.add_argument("--dry_run", action="store_true", help="Generate config without training")
    misc_group.add_argument("--config_only", action="store_true", help="Only save configs")
    misc_group.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path (default: project SuperBPE-56000; see umich_slm.tokenizer_defaults)")

    args = parser.parse_args()

    # Set random seeds for reproducibility
    from umich_slm.utils import set_seed
    set_seed(args.seed, deterministic=args.deterministic)
    print(f"Random seed: {args.seed}" + (" (deterministic mode)" if args.deterministic else ""))

    # Override logging_steps if wandb_log_every_step is enabled
    if args.wandb_log_every_step:
        if not args.wandb_project:
            print("Warning: --wandb_log_every_step requires --wandb_project. Ignoring --wandb_log_every_step.")
        else:
            args.logging_steps = 1
            print("Setting logging_steps=1 to record every step in WandB")

    # Validate val_percent
    if args.val_percent < 0 or args.val_percent >= 100:
        raise ValueError(f"val_percent must be between 0 and 100, got {args.val_percent}")

    # Calculate training steps based on token target
    train_tokens, val_tokens, max_steps, max_samples = calculate_training_steps(
        target_tokens_b=args.target_tokens_b,
        val_percent=args.val_percent,
        cutoff_len=args.cutoff_len,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_gpus=args.num_gpus,
    )

    total_tokens = train_tokens + val_tokens

    print("=" * 60)
    print("Pretraining Configuration")
    print("=" * 60)
    print(f"Total tokens:      {total_tokens:,} ({total_tokens / 1e9:.2f}B)")
    print(f"Training tokens:   {train_tokens:,} ({train_tokens / 1e9:.2f}B, {100 - args.val_percent:.1f}%%)")
    print(f"Validation tokens: {val_tokens:,} ({val_tokens / 1e9:.2f}B, {args.val_percent:.1f}%%, held out)")
    print(f"Sequence length:   {args.cutoff_len}")
    print(f"Batch size:        {args.batch_size} (per device)")
    print(f"Grad accum steps:  {args.gradient_accumulation_steps}")
    print(f"Effective batch:   {args.batch_size * args.gradient_accumulation_steps * args.num_gpus} samples")
    print(f"Tokens per step:   {args.batch_size * args.gradient_accumulation_steps * args.num_gpus * args.cutoff_len:,}")
    print(f"Max steps:         {max_steps:,}")
    print(f"Max samples:       {max_samples:,} (first {max_samples:,} samples from training dataset)")
    print(f"Streaming:         {args.streaming}")
    print(f"Seed:              {args.seed}")
    if not args.streaming:
        print(f"Reproducibility:   Enabled (same {max_samples:,} training samples used each run)")
    print("-" * 60)
    print(f"Dataset:           {args.dataset}")
    if args.dataset_dir:
        print(f"Dataset dir:       {args.dataset_dir}")
    else:
        print(f"Dataset dir:       (default: configs/llama_factory)")
    if args.do_eval and args.eval_dataset:
        print(f"Eval dataset:      {args.eval_dataset}")
        print(f"Eval steps:        {args.eval_steps}")
    else:
        print(f"Evaluation:        Disabled")
    print("=" * 60)

    # Generate run name
    run_name = args.run_name or get_run_name(args)
    print(f"Run name: {run_name}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine tokenizer and get vocab_size
    tokenizer_path = args.tokenizer_path or "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    print(f"\n=== Loading Tokenizer ===")
    print(f"Tokenizer: {tokenizer_path}")
    vocab_size = get_vocab_size_from_tokenizer(tokenizer_path)
    print(f"Vocabulary size: {vocab_size:,}")

    # Create model configuration with vocab_size from tokenizer
    print("\n=== Creating Model Configuration ===")
    config_dict = create_model_config(args, vocab_size=vocab_size)
    print(f"Model size: {args.size}")

    # Delphi recipe run-context: requires max_steps which is now known.
    if getattr(args, "delphi_recipe", False):
        global_batch = args.batch_size * args.gradient_accumulation_steps * args.num_gpus
        run_tokens = train_tokens
        config_dict["delphi_run_tokens"] = float(run_tokens)
        config_dict["delphi_run_batch"] = int(global_batch)
        config_dict["delphi_run_total_steps"] = int(max_steps)
        print(
            f"Delphi recipe: T={run_tokens:.2e}, B={global_batch}, steps={max_steps}, "
            f"opt={args.delphi_optimizer}, sched={args.delphi_schedule}"
        )

    # Save initialized model
    model_path = output_dir / "models" / run_name

    # Check if model already exists and warn if vocab_size might be wrong
    if model_path.exists() and (model_path / "config.json").exists():
        import json
        existing_config_path = model_path / "config.json"
        with open(existing_config_path, "r") as f:
            existing_config = json.load(f)
        existing_vocab_size = existing_config.get("vocab_size", None)
        if existing_vocab_size is not None and existing_vocab_size != vocab_size:
            print(f"\n⚠️  WARNING: Existing model at {model_path} has vocab_size {existing_vocab_size}")
            print(f"   But tokenizer has vocab_size {vocab_size}")
            print(f"   The existing model will be overwritten with correct vocab_size")
            print(f"   If you want to keep the existing model, rename or move it first.\n")

    if not args.config_only:
        print(f"\n=== Saving Model to {model_path} ===")
        # Determine dtype based on args (bf16 takes precedence over fp16)
        if args.bf16:
            dtype = "bf16"
        elif args.fp16:
            dtype = "fp16"
        else:
            dtype = None
        save_model_init(config_dict, model_path, tokenizer_path=args.tokenizer_path or tokenizer_path, dtype=dtype)
    else:
        print(f"\n=== Skipping model initialization (--config_only) ===")
        model_path.mkdir(parents=True, exist_ok=True)
        from umich_slm import UMichSLMConfig
        config = UMichSLMConfig(**config_dict)
        config.save_pretrained(model_path)

    # Generate LLaMA Factory config
    print(f"\n=== Generating LLaMA Factory Config ===")
    lf_config = generate_llamafactory_config(args, model_path, run_name, max_steps, max_samples)

    config_path = output_dir / "configs" / f"{run_name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(lf_config, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to {config_path}")

    # Generate WandB tags and group
    wandb_tags = get_wandb_tags(args) if args.wandb_project else None
    wandb_group = get_wandb_group(args) if args.wandb_project else None

    # Run training
    if not args.dry_run and not args.config_only:
        print(f"\n=== Starting Training ===")
        return_code = run_training(
            config_path,
            dry_run=False,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity if args.wandb_project else None,
            wandb_tags=wandb_tags,
            wandb_group=wandb_group,
        )
        sys.exit(return_code)
    else:
        print(f"\n=== Dry Run Complete ===")
        if wandb_tags:
            print(f"\nWandB Tags: {', '.join(wandb_tags)}")
        if wandb_group:
            print(f"WandB Group: {wandb_group}")
        print(f"\nTo run training, execute:")
        env_vars = []
        if args.wandb_project:
            env_vars.append(f"WANDB_PROJECT={args.wandb_project}")
        if args.wandb_entity and args.wandb_project:
            env_vars.append(f"WANDB_ENTITY={args.wandb_entity}")
        if wandb_tags:
            env_vars.append(f"WANDB_TAGS='{','.join(wandb_tags)}'")
        if wandb_group:
            env_vars.append(f"WANDB_RUN_GROUP={wandb_group}")
        env_prefix = " ".join(env_vars) + " " if env_vars else ""
        print(f"  {env_prefix}python scripts/llamafactory_train.py train {config_path}")


if __name__ == "__main__":
    main()
