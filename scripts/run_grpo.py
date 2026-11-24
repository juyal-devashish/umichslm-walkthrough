#!/usr/bin/env python3
"""GRPO (Group Relative Policy Optimization) training on verifiable math rewards.

Starts from an SFT checkpoint and fine-tunes with two rule-based reward signals:
  - format_reward:   compliance with the <think>/<answer> 6-token schema
  - accuracy_reward: exact-match of the extracted <answer> against the gold answer

Uses trl.GRPOTrainer directly (analogous to how pretrain_packed.py uses
transformers.Trainer directly) so that custom reward functions are easy to wire in.

Example usage (single run):
    python scripts/run_grpo.py \\
        --model_path ./outputs/umichslm-sft-lr2e-5 \\
        --train_data /path/to/data \\
        --val_data   /path/to/data \\
        --output_dir ./outputs/umichslm-grpo-gsm8k

With YAML config + CLI overrides (used by the SLURM sweep):
    python scripts/run_grpo.py \\
        --config configs/llama_factory/umichslm_grpo.yaml \\
        --model_path /path/to/data \\
        --num_generations 8 \\
        --epsilon 0.2
"""

import argparse
import sys
from pathlib import Path

import torch
import wandb
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Register custom UMichSLM model classes before any AutoModel call.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import umich_slm  # noqa: F401

from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent))
from grpo_data.reward_fns import accuracy_reward, format_reward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_dataset(jsonl_path: str, split_name: str):
    return load_dataset("json", data_files={split_name: jsonl_path})[split_name]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPO training")
    parser.add_argument("--config",          type=str, default="configs/llama_factory/umichslm_grpo.yaml")
    parser.add_argument("--model_path",      type=str, help="SFT checkpoint to start from")
    parser.add_argument("--train_data",      type=str, help="Path to train .jsonl")
    parser.add_argument("--val_data",        type=str, help="Path to val .jsonl")
    parser.add_argument("--output_dir",      type=str)
    parser.add_argument("--num_generations", type=int,   help="G: completions sampled per prompt")
    parser.add_argument("--learning_rate",   type=float)
    parser.add_argument("--epsilon",         type=float, help="PPO-style clipping ratio")
    parser.add_argument("--beta",            type=float, help="KL penalty coefficient")
    parser.add_argument("--run_name",        type=str,   default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # CLI flags override YAML values.
    for flag, key in [
        (args.model_path,      "model_name_or_path"),
        (args.train_data,      "train_data"),
        (args.val_data,        "val_data"),
        (args.output_dir,      "output_dir"),
        (args.num_generations, "num_generations"),
        (args.learning_rate,   "learning_rate"),
        (args.epsilon,         "epsilon"),
        (args.beta,            "beta"),
    ]:
        if flag is not None:
            cfg[key] = flag

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    G   = cfg.get("num_generations", 8)
    eps = cfg.get("epsilon", 0.2)
    run_name = args.run_name or f"grpo-g{G}-eps{eps}"

    wandb.init(
        project="umich-slm",
        name=run_name,
        config=cfg,
    )

    # ------------------------------------------------------------------
    # Model + tokenizer
    # ------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Datasets — each row must have "prompt" (str) and "answer" (str).
    # ------------------------------------------------------------------
    train_dataset = _build_dataset(cfg["train_data"], "train")
    eval_dataset  = _build_dataset(cfg["val_data"], "val") if cfg.get("val_data") else None

    # ------------------------------------------------------------------
    # GRPO config
    # ------------------------------------------------------------------
    grpo_cfg = GRPOConfig(
        output_dir=cfg["output_dir"],
        # Generation
        num_generations=G,
        temperature=cfg.get("temperature", 1.0),
        max_completion_length=cfg.get("max_new_tokens", 512),
        max_prompt_length=cfg.get("max_prompt_length", 512),
        # Policy gradient
        beta=cfg.get("beta", 0.001),
        epsilon=cfg.get("epsilon", 0.2),
        # Optimiser
        learning_rate=cfg.get("learning_rate", 1e-6),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        # Memory
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        # Logging / saving
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 200),
        eval_steps=cfg.get("eval_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 3),
        report_to=cfg.get("report_to", "wandb"),
        run_name=run_name,
        seed=cfg.get("seed", 42),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
    )

    # ------------------------------------------------------------------
    # Trainer — reward_funcs are called in order; their scalar outputs
    # are summed to produce the final reward signal per completion.
    # ------------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, accuracy_reward],
        args=grpo_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    wandb.finish()


if __name__ == "__main__":
    main()
