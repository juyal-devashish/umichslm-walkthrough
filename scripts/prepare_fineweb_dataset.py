#!/usr/bin/env python3
"""Prepare fineweb-edu dataset for pretraining.

This script downloads and prepares a subset of the HuggingFaceFW/fineweb-edu
dataset for pretraining, creating separate train and validation splits.

The prepared dataset is saved locally so training runs can use it directly
without re-downloading or re-processing each time.

Default configuration:
    - Training set: ~1B tokens (~488K samples at 2048 avg tokens/sample)
    - Validation set: 3% of training tokens (~30M tokens, ~15K samples)
    - Format: Arrow format (efficient for training, zero-copy reads)

Example usage:
    # Prepare dataset with defaults (1B train, 3% validation = ~30M tokens)
    python scripts/prepare_fineweb_dataset.py --output_dir ./data/fineweb-edu-1B

    # Custom token counts
    python scripts/prepare_fineweb_dataset.py \
        --train_tokens 500_000_000 \
        --val_tokens 5_000_000 \
        --output_dir ./data/fineweb-edu-500M

    # Use custom validation percentage
    python scripts/prepare_fineweb_dataset.py \
        --train_tokens 500_000_000 \
        --val_percent 5.0 \
        --output_dir ./data/fineweb-edu-500M

    # Specify sequence length for sample calculation
    python scripts/prepare_fineweb_dataset.py \
        --output_dir ./data/fineweb-edu-1B \
        --seq_length 4096

    # Dry run to see what would be done
    python scripts/prepare_fineweb_dataset.py --dry_run
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm
import pyarrow as pa
import pyarrow.ipc as ipc


def estimate_samples_for_tokens(
    target_tokens: int,
    avg_tokens_per_sample: int = 2048,
    buffer_factor: float = 1.05,
) -> int:
    """Estimate number of samples needed for target token count.

    Args:
        target_tokens: Target number of tokens.
        avg_tokens_per_sample: Average tokens per sample (conservative estimate).
        buffer_factor: Multiplier to ensure we have enough samples.

    Returns:
        Estimated number of samples needed.
    """
    # Add buffer to account for variation in sample lengths
    return int((target_tokens / avg_tokens_per_sample) * buffer_factor)


def count_tokens_in_samples(
    dataset: Dataset,
    tokenizer,
    num_samples: Optional[int] = None,
    text_column: str = "text",
) -> tuple[int, float]:
    """Count tokens in dataset samples.

    Args:
        dataset: HuggingFace dataset.
        tokenizer: Tokenizer to use for counting.
        num_samples: Number of samples to count (None for all).
        text_column: Name of the text column.

    Returns:
        Tuple of (total_tokens, avg_tokens_per_sample).
    """
    total_tokens = 0
    samples_to_check = min(num_samples or len(dataset), len(dataset))

    for i in tqdm(range(samples_to_check), desc="Counting tokens"):
        text = dataset[i][text_column]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(tokens)

    avg_tokens = total_tokens / samples_to_check if samples_to_check > 0 else 0
    return total_tokens, avg_tokens


def prepare_dataset(
    train_tokens: int,
    val_tokens: int,
    output_dir: Path,
    seq_length: int = 2048,
    seed: int = 42,
    tokenizer_name: str = "meta-llama/Meta-Llama-3-8B",
    verify_tokens: bool = True,
    dry_run: bool = False,
) -> dict:
    """Prepare fineweb-edu dataset with train/val splits.

    Args:
        train_tokens: Target tokens for training set.
        val_tokens: Target tokens for validation set.
        output_dir: Directory to save prepared dataset.
        seq_length: Expected sequence length (for sample estimation).
        seed: Random seed for reproducibility.
        tokenizer_name: Tokenizer to use for token counting.
        verify_tokens: Whether to verify token counts after preparation.
        dry_run: If True, only print what would be done.

    Returns:
        Dictionary with dataset statistics.
    """
    output_dir = Path(output_dir)

    # Estimate samples needed (using seq_length as conservative avg)
    # fineweb-edu samples are often longer than 2048 tokens
    train_samples = estimate_samples_for_tokens(train_tokens, seq_length)
    val_samples = estimate_samples_for_tokens(val_tokens, seq_length)
    total_samples = train_samples + val_samples

    print("=" * 60)
    print("Fineweb-edu Dataset Preparation")
    print("=" * 60)
    print(f"Training tokens target:   {train_tokens:,} ({train_tokens/1e9:.2f}B)")
    print(f"Validation tokens target: {val_tokens:,} ({val_tokens/1e6:.1f}M)")
    print(f"Sequence length:          {seq_length}")
    print(f"Estimated train samples:  {train_samples:,}")
    print(f"Estimated val samples:    {val_samples:,}")
    print(f"Total samples to load:    {total_samples:,}")
    print(f"Output directory:         {output_dir}")
    print(f"Random seed:              {seed}")
    print("=" * 60)

    if dry_run:
        print("\n[DRY RUN] Would prepare dataset with above configuration.")
        return {"dry_run": True}

    # Load tokenizer for verification
    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Load dataset from HuggingFace
    print("\nLoading fineweb-edu dataset from HuggingFace...")
    print("(This may take a while for first download)")

    # Use streaming to avoid loading entire dataset
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # Take required samples
    print(f"\nTaking first {total_samples:,} samples...")
    samples = []
    for i, sample in enumerate(tqdm(dataset, total=total_samples, desc="Loading samples")):
        if i >= total_samples:
            break
        samples.append({"text": sample["text"]})

    print(f"Loaded {len(samples):,} samples")

    # Create dataset from samples
    full_dataset = Dataset.from_list(samples)

    # Split into train and validation
    print(f"\nSplitting dataset (train: {train_samples:,}, val: {val_samples:,})...")

    # Take first train_samples for training, next val_samples for validation
    # This ensures reproducibility - same samples every time
    train_dataset = full_dataset.select(range(train_samples))
    val_dataset = full_dataset.select(range(train_samples, train_samples + val_samples))

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
    })

    # Verify token counts if requested
    stats = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "target_train_tokens": train_tokens,
        "target_val_tokens": val_tokens,
        "seq_length": seq_length,
        "seed": seed,
    }

    if verify_tokens:
        print("\nVerifying token counts (sampling 1000 examples)...")

        # Sample to estimate actual token counts
        sample_size = min(1000, len(train_dataset))
        _, avg_tokens = count_tokens_in_samples(
            train_dataset.select(range(sample_size)),
            tokenizer,
            sample_size,
        )

        estimated_train_tokens = int(len(train_dataset) * avg_tokens)
        estimated_val_tokens = int(len(val_dataset) * avg_tokens)

        stats["avg_tokens_per_sample"] = avg_tokens
        stats["estimated_train_tokens"] = estimated_train_tokens
        stats["estimated_val_tokens"] = estimated_val_tokens

        print(f"Average tokens per sample: {avg_tokens:.1f}")
        print(f"Estimated train tokens:    {estimated_train_tokens:,} ({estimated_train_tokens/1e9:.2f}B)")
        print(f"Estimated val tokens:      {estimated_val_tokens:,} ({estimated_val_tokens/1e6:.1f}M)")

    # Save dataset
    print(f"\nSaving dataset to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset directory
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save in Arrow format as single files to avoid mixed file type issues
    # Arrow format provides zero-copy reads and is faster for training workloads
    # We save each split as a single Arrow file to work with LLaMA Factory's loader
    train_arrow_path = dataset_dir / "train.arrow"
    val_arrow_path = dataset_dir / "validation.arrow"

    # Convert HuggingFace dataset to Arrow table and save as single file
    # HuggingFace datasets store data in Arrow format internally
    # Access the underlying Arrow table from the dataset's data attribute
    train_table = train_dataset.data.table
    val_table = val_dataset.data.table

    # Write Arrow IPC files (single file format)
    # This creates a single .arrow file that LLaMA Factory can load without mixed file type issues
    with pa.OSFile(str(train_arrow_path), 'wb') as sink:
        with ipc.new_file(sink, train_table.schema) as writer:
            writer.write_table(train_table)
    file_size_mb = train_arrow_path.stat().st_size / 1024 / 1024
    print(f"Saved train dataset to {train_arrow_path} ({file_size_mb:.1f} MB)")

    with pa.OSFile(str(val_arrow_path), 'wb') as sink:
        with ipc.new_file(sink, val_table.schema) as writer:
            writer.write_table(val_table)
    file_size_mb = val_arrow_path.stat().st_size / 1024 / 1024
    print(f"Saved validation dataset to {val_arrow_path} ({file_size_mb:.1f} MB)")

    # Save statistics
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    # Create dataset_info.json entry for LLaMA Factory
    # For single Arrow files, point file_name to the .arrow file and specify format
    dataset_info = {
        "fineweb_edu_1B_train": {
            "file_name": str(train_arrow_path),
            "format": "arrow",
            "columns": {
                "prompt": "text"
            }
        },
        "fineweb_edu_1B_val": {
            "file_name": str(val_arrow_path),
            "format": "arrow",
            "columns": {
                "prompt": "text"
            }
        }
    }

    dataset_info_path = output_dir / "dataset_info.json"
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Saved dataset_info.json to {dataset_info_path}")

    # Print usage instructions
    print("\n" + "=" * 60)
    print("Dataset prepared successfully!")
    print("=" * 60)
    print("\nTo use this dataset with LLaMA Factory, either:")
    print("\n1. Copy entries from dataset_info.json to configs/llama_factory/dataset_info.json")
    print("\n2. Or use --dataset_dir flag:")
    print(f"   --dataset_dir {output_dir}")
    print("\nDataset names:")
    print("   - fineweb_edu_1B_train (for training)")
    print("   - fineweb_edu_1B_val (for validation)")
    print("\nExample training command:")
    print(f"   python scripts/pretrain_fineweb.py --dataset fineweb_edu_1B_train \\")
    print(f"       --dataset_dir {output_dir} --do_eval --eval_dataset fineweb_edu_1B_val")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare fineweb-edu dataset for pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/fineweb-edu-1B",
        help="Directory to save prepared dataset",
    )
    parser.add_argument(
        "--train_tokens",
        type=int,
        default=1_000_000_000,
        help="Target number of tokens for training set",
    )
    parser.add_argument(
        "--val_tokens",
        type=int,
        default=None,
        help="Target number of tokens for validation set (overrides --val_percent if set)",
    )
    parser.add_argument(
        "--val_percent",
        type=float,
        default=3.0,
        help="Validation set size as percentage of training tokens (default: 3.0%%)",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
        help="Expected sequence length (for sample estimation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Tokenizer to use for token counting (default: LLaMA 3)",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip token count verification",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without actually preparing dataset",
    )

    args = parser.parse_args()

    # Calculate validation tokens from percentage if not explicitly provided
    if args.val_tokens is None:
        val_tokens = int(args.train_tokens * args.val_percent / 100.0)
        print(f"Calculated validation tokens: {val_tokens:,} ({args.val_percent}%% of {args.train_tokens:,} training tokens)")
    else:
        val_tokens = args.val_tokens
        if args.val_percent != 3.0:
            print(f"Warning: --val_tokens is set, ignoring --val_percent={args.val_percent}")

    stats = prepare_dataset(
        train_tokens=args.train_tokens,
        val_tokens=val_tokens,
        output_dir=Path(args.output_dir),
        seq_length=args.seq_length,
        seed=args.seed,
        tokenizer_name=args.tokenizer,
        verify_tokens=not args.no_verify,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print("\nDataset statistics:")
        for key, value in stats.items():
            if isinstance(value, int) and value > 1000:
                print(f"  {key}: {value:,}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
