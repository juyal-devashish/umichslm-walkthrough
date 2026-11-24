#!/usr/bin/env python3
"""Pretrain on packed Arrow datasets produced by prepare_mix_dataset.py --pack.

Each Arrow record contains a single 'input_ids' list of exactly seq_length
tokens.  No tokenization happens at training time; DataCollatorForLanguageModeling
creates labels by copying input_ids (standard CLM next-token-prediction loss).

This script uses HuggingFace Trainer directly rather than LLaMA Factory because
LLaMA Factory's pretraining stage expects raw text, not pre-tokenized sequences.
Packing already eliminates padding, so the effective batch size equals
per_device_train_batch_size * gradient_accumulation_steps * num_gpus * seq_length tokens.

Typical workflow
----------------
1. Initialize a model with pretrain_fineweb.py (config + random weights):
       python scripts/pretrain_fineweb.py --size small --dry_run --output_dir ./outputs
   Then remove --dry_run to actually save the model config, or use:
       python -c "
       from umich_slm import UMichSLMConfig, UMichSLMForCausalLM, get_umichslm_config
       from transformers import AutoTokenizer
       cfg = get_umichslm_config('small')
       model = UMichSLMForCausalLM(cfg)
       model.save_pretrained('./outputs/models/small')
       tok = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
       tok.save_pretrained('./outputs/models/small')
       "

2. Prepare a packed dataset:
       python scripts/prepare_mix_dataset.py --pack --output_dir ./data/packed

3. Train:
       python scripts/pretrain_packed.py \
           --model_path ./outputs/models/small \
           --train_file ./data/packed/dataset/train_packed.arrow \
           --val_file   ./data/packed/dataset/validation_packed.arrow \
           --output_dir ./outputs/packed-pretrain

Example usage
-------------
    # Basic run
    python scripts/pretrain_packed.py \
        --model_path ./outputs/models/small \
        --train_file ./data/packed/dataset/train_packed.arrow \
        --output_dir ./outputs/packed-pretrain

    # With validation and WandB
    python scripts/pretrain_packed.py \
        --model_path ./outputs/models/small \
        --train_file  ./data/packed/dataset/train_packed.arrow \
        --val_file    ./data/packed/dataset/validation_packed.arrow \
        --output_dir  ./outputs/packed-pretrain \
        --wandb_project umichslm-pretrain \
        --wandb_name    small-packed-1B

    # Custom hyperparameters
    python scripts/pretrain_packed.py \
        --model_path ./outputs/models/small \
        --train_file ./data/packed/dataset/train_packed.arrow \
        --output_dir ./outputs/packed-pretrain \
        --batch_size 16 --grad_accum 8 --learning_rate 3e-4 --max_steps 50000
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain on packed Arrow datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Path to an initialized UMichSLM model directory (config + weights).",
    )
    parser.add_argument(
        "--train_file", required=True,
        help="Path to train_packed.arrow produced by prepare_mix_dataset.py --pack.",
    )
    parser.add_argument(
        "--val_file", default=None,
        help="Optional path to validation_packed.arrow. Enables per-step eval.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to save checkpoints and the final model.",
    )

    # Hyperparameters
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--grad_accum",   type=int,   default=16,
                        help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_steps",    type=int,   default=-1,
                        help="Hard step limit. -1 means run for num_train_epochs.")
    parser.add_argument("--num_epochs",   type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str,   default="cosine")

    # Logging / checkpointing
    parser.add_argument("--logging_steps", type=int,  default=50)
    parser.add_argument("--save_steps",    type=int,  default=1000)
    parser.add_argument("--eval_steps",    type=int,  default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)

    # WandB
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project name. Omit to disable WandB.")
    parser.add_argument("--wandb_entity",  type=str, default=None)
    parser.add_argument("--wandb_name",    type=str, default=None,
                        help="WandB run name.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_workers", type=int, default=4)

    # Token-Superposition Training (TST)
    parser.add_argument("--tst_enabled", action="store_true",
                        help="Enable TST phase (i) + recovery (Peng et al., arXiv:2605.06546).")
    parser.add_argument("--tst_bag_size", type=int, default=6,
                        help="TST bag size s (paper recommends 4-8 for dense ≤3B).")
    parser.add_argument("--tst_recovery_ratio", type=float, default=0.3,
                        help="Fraction of total steps spent in TST phase (i) before recovery.")
    parser.add_argument("--tst_loss_weighting", type=str, default="uniform",
                        choices=["uniform", "power_law"],
                        help="Within-bag weighting for MCE (power_law per Appendix D for s>=8).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure WandB before importing Trainer so it picks up env vars
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
    report_to = "wandb" if args.wandb_project else "tensorboard"

    # ------------------------------------------------------------------ model
    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    # ------------------------------------------------------------------ data
    print(f"Loading packed train dataset from {args.train_file}")
    data_files: dict[str, str] = {"train": args.train_file}
    if args.val_file:
        print(f"Loading packed val dataset from {args.val_file}")
        data_files["validation"] = args.val_file

    # load_dataset("arrow") reads Arrow IPC files produced by _write_packed_arrow().
    # Each record has a single "input_ids" column (list<int32>, length == seq_length).
    dataset = load_dataset("arrow", data_files=data_files)
    print(f"  train: {len(dataset['train']):,} packed sequences")
    if args.val_file:
        print(f"  val:   {len(dataset['validation']):,} packed sequences")

    # Infer seq_length from the first record so we can report tokens/step
    seq_length: int = len(dataset["train"][0]["input_ids"])
    effective_batch = args.batch_size * args.grad_accum
    tokens_per_step = effective_batch * seq_length
    print(f"  seq_length:       {seq_length}")
    print(f"  effective batch:  {effective_batch} seqs = {tokens_per_step:,} tokens/step")

    # DataCollatorForLanguageModeling with mlm=False:
    #   - pads input_ids to the longest in the batch (no-op since all are seq_length)
    #   - sets labels = input_ids (HF CausalLM models shift internally)
    #   - adds attention_mask of all 1s (no padding tokens in packed sequences)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --------------------------------------------------------- TrainingArguments
    eval_strategy = "steps" if args.val_file else "no"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        # batch / steps
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs,

        # optimiser
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim="adamw_torch",

        # precision / memory
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        gradient_checkpointing=True,

        # logging / saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps if args.val_file else None,
        evaluation_strategy=eval_strategy,
        load_best_model_at_end=bool(args.val_file),
        metric_for_best_model="eval_loss" if args.val_file else None,
        greater_is_better=False,

        # misc
        report_to=report_to,
        run_name=args.wandb_name,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
    )

    # ----------------------------------------------------------------- Trainer
    callbacks = []
    if getattr(args, "tst_enabled", False):
        from umich_slm.callbacks import TSTTrainerCallback
        # Propagate TST settings into the model config so forward() picks them up.
        model.config.tst_enabled = True
        model.config.tst_bag_size = int(args.tst_bag_size)
        model.config.tst_loss_weighting = args.tst_loss_weighting
        callbacks.append(
            TSTTrainerCallback(
                enabled=True,
                bag_size=int(args.tst_bag_size),
                recovery_ratio=float(args.tst_recovery_ratio),
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks or None,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
