#!/usr/bin/env python3
"""Prepare mixed pretraining dataset from local sources.

Loads web, code, textbook, math, and math_reasoning data from local directories,
samples each according to a mixture ratio, and saves train/val Arrow splits.

Default configuration:
    - Training set: ~1B tokens
    - Validation set: 3% of training tokens (~30M tokens)
    - Mixture: web 60%, textbook 20%, code 10%, math 5%, math_reasoning 5%
    - Format: Arrow IPC format (efficient for training, zero-copy reads)
    - Tokenizer: meta-llama/Meta-Llama-3-8B

Example usage:
    # Prepare dataset with defaults (1B train, 3% validation)
    python scripts/prepare_mix_dataset.py --output_dir ./data/build/1B

    # Custom token count and mixture
    python scripts/prepare_mix_dataset.py \
        --train_tokens 200_000_000 \
        --output_dir ./data/build/200M \
        --mixture_ratio '{"math":0.2,"math_reasoning":0.2,"code":0.1,"textbook":0.2,"web":0.3}'

    # Sequence packing: tokenize + pack into fixed-length sequences (no padding waste)
    # Outputs input_ids Arrow files; train with scripts/pretrain_packed.py
    python scripts/prepare_mix_dataset.py \
        --output_dir ./data/build/1B_packed \
        --pack

    # Dry run to see what would be done
    python scripts/prepare_mix_dataset.py --dry_run
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from datasets import load_dataset, Dataset, DatasetDict, interleave_datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import pyarrow as pa
import pyarrow.ipc as ipc



def setup_tokenizer(tokenizer_name="meta-llama/Meta-Llama-3-8B"):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True
    )

    # Use eos as pad (standard for Llama)
    tokenizer.pad_token = tokenizer.eos_token

    special_tokens_dict = {
        "additional_special_tokens": [
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|end|>",
            "<think>", "</think>",
            "<retrieve>", "</retrieve>",
            "<result>", "</result>",
            "<answer>", "</answer>",
            "<code>", "</code>",
            "<cite>", "</cite>",
        ]
    }

    tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer


def pack_sequences(
    token_id_lists: list[list[int]],
    eos_token_id: int,
    seq_length: int,
    seed: int = 42,
) -> tuple[list[list[int]], dict]:
    """Pack variable-length token-id lists into fixed-length sequences.

    Documents are shuffled, then concatenated with EOS separators and chunked
    into seq_length blocks. The trailing partial block is discarded to ensure
    uniform sequence lengths. Cross-document attention is allowed within each
    packed sequence, which is standard for pretraining (the model learns EOS
    as a document boundary signal).

    Args:
        token_id_lists: Pre-tokenized documents as lists of integer token ids.
        eos_token_id: Token id inserted between consecutive documents.
        seq_length: Output sequence length; every returned sequence has exactly
            this many tokens.
        seed: RNG seed for document shuffle order.

    Returns:
        (sequences, stats) where sequences is a list of int lists, each
        exactly seq_length tokens long.
    """
    rng = random.Random(seed)
    order = list(range(len(token_id_lists)))
    rng.shuffle(order)

    buffer: list[int] = []
    packed: list[list[int]] = []
    total_doc_tokens = 0

    for idx in tqdm(order, desc="  packing docs", leave=False):
        ids = token_id_lists[idx]
        total_doc_tokens += len(ids)
        if buffer:
            buffer.append(eos_token_id)
        buffer.extend(ids)
        while len(buffer) >= seq_length:
            packed.append(buffer[:seq_length])
            buffer = buffer[seq_length:]

    stats = {
        "num_packed_sequences": len(packed),
        "num_input_docs": len(token_id_lists),
        "total_doc_tokens": total_doc_tokens,
        "packed_tokens": len(packed) * seq_length,
        "discarded_tokens": len(buffer),
        "utilization_pct": round(
            len(packed) * seq_length / max(total_doc_tokens, 1) * 100, 1
        ),
    }
    return packed, stats


def _write_packed_arrow(sequences: list[list[int]], path: Path) -> None:
    """Write a list of fixed-length token-id sequences to an Arrow IPC file."""
    schema = pa.schema([pa.field("input_ids", pa.list_(pa.int32()))])
    ids_col = pa.array(sequences, type=pa.list_(pa.int32()))
    table = pa.table({"input_ids": ids_col}, schema=schema)
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def _pack_and_save(
    full_token_ids: list[list[int]],
    train_split_idx: int,
    eos_token_id: int,
    seq_length: int,
    seed: int,
    output_dir: Path,
    train_tokens: int,
    val_tokens: int,
    mixture_ratio: dict,
) -> dict:
    """Pack pre-tokenized documents and save as fixed-length Arrow sequences."""
    if eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; cannot insert document separators.")

    train_ids_list = full_token_ids[:train_split_idx]
    val_ids_list   = full_token_ids[train_split_idx:]

    print("\nPacking train sequences...")
    train_seqs, train_stats = pack_sequences(train_ids_list, eos_token_id, seq_length, seed)
    print(
        f"  {train_stats['num_packed_sequences']:,} packed sequences "
        f"({train_stats['utilization_pct']}% utilization, "
        f"{train_stats['discarded_tokens']:,} tokens discarded)"
    )

    print("Packing val sequences...")
    val_seqs, val_stats = pack_sequences(val_ids_list, eos_token_id, seq_length, seed + 1)
    print(
        f"  {val_stats['num_packed_sequences']:,} packed sequences "
        f"({val_stats['utilization_pct']}% utilization, "
        f"{val_stats['discarded_tokens']:,} tokens discarded)"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_path = dataset_dir / "train_packed.arrow"
    val_path   = dataset_dir / "validation_packed.arrow"

    print(f"\nSaving packed dataset to {output_dir}...")
    _write_packed_arrow(train_seqs, train_path)
    print(f"Saved packed train to {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")

    _write_packed_arrow(val_seqs, val_path)
    print(f"Saved packed val to   {val_path} ({val_path.stat().st_size / 1024 / 1024:.1f} MB)")

    stats = {
        "packed": True,
        "seq_length": seq_length,
        "seed": seed,
        "mixture_ratio": mixture_ratio,
        "target_train_tokens": train_tokens,
        "target_val_tokens": val_tokens,
        "train": train_stats,
        "val": val_stats,
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    print("\n" + "=" * 60)
    print("Packed dataset ready!")
    print("=" * 60)
    print("\nTrain with pre-tokenized packed sequences:")
    print(f"   python scripts/pretrain_packed.py \\")
    print(f"       --model_path <initialized-model-dir> \\")
    print(f"       --train_file {train_path} \\")
    print(f"       --val_file {val_path} \\")
    print(f"       --output_dir ./outputs/packed-pretrain")

    return stats


def prepare_dataset(
    train_tokens: int,
    val_tokens: int,
    output_dir: Path,
    seq_length: int = 2048,
    mix_input_dir: dict = {
        "web": "./data/fineweb",
        "code": "./data/code",
        "textbook": "./data/textbook",
        "math": "./data/math",
        "math_reasoning": "./data/math_reasoning",
    },
    mixture_ratio: dict = {'math': 0.05, 'code':0.1, 'math_reasoning': 0.05, 'textbook': 0.20, 'web': 0.60},
    seed: int = 42,
    tokenizer_name: str = "meta-llama/Meta-Llama-3-8B",
    dry_run: bool = False,
    pack: bool = False,
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
    total_tokens = train_tokens + val_tokens

    print("=" * 60)
    print("Fineweb-edu Dataset Preparation")
    print("=" * 60)
    print(f"Training tokens target:   {train_tokens:,} ({train_tokens/1e9:.2f}B)")
    print(f"Validation tokens target: {val_tokens:,} ({val_tokens/1e6:.1f}M)")
    print(f"Sequence length:          {seq_length}")
    print(f"Output directory:         {output_dir}")
    print(f"Random seed:              {seed}")
    print(f"Mixture: {mixture_ratio}")
    print(f"Sequence packing:         {'enabled' if pack else 'disabled'}")
    print("=" * 60)

    if dry_run:
        print("\n[DRY RUN] Would prepare dataset with above configuration.")
        return {"dry_run": True}

    # Load tokenizer for verification
    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = setup_tokenizer(tokenizer_name)

    # Load streaming dataset from given input dir (streaming=True avoids loading entire datasets into RAM)
    web_dataset = load_dataset("parquet", data_dir=mix_input_dir["web"], split="train", streaming=True)
    textbook_dataset = load_dataset("json", data_dir=mix_input_dir["textbook"], split="train", streaming=True)
    math_reasoning_dataset = load_dataset("json", data_dir=mix_input_dir["math_reasoning"], split="train", streaming=True)
    math_dataset = load_dataset("json", data_dir=mix_input_dir["math"], split="train", streaming=True)
    code_dataset = load_dataset("json", data_dir=mix_input_dir["code"], split="train", streaming=True)

    # Normalize all datasets to {"text": ...} only — schemas differ across domains
    # (textbook uses "prompt" instead of "text"; fineweb has extra metadata columns)
    web_dataset            = web_dataset.map(lambda x: {"text": x["text"]},   remove_columns=list(web_dataset.features.keys()))
    textbook_dataset       = textbook_dataset.map(lambda x: {"text": x["text"]}, remove_columns=list(textbook_dataset.features.keys()))
    math_dataset           = math_dataset.map(lambda x: {"text": x["text"]},   remove_columns=list(math_dataset.features.keys()))
    math_reasoning_dataset = math_reasoning_dataset.map(lambda x: {"text": x["text"]}, remove_columns=list(math_reasoning_dataset.features.keys()))
    code_dataset           = code_dataset.map(lambda x: {"text": x["text"]},   remove_columns=list(code_dataset.features.keys()))

    # Interleave datasets using mixture probabilities (list order must match dataset order)
    full_dataset = interleave_datasets(
        [web_dataset, textbook_dataset, math_reasoning_dataset, math_dataset, code_dataset],
        probabilities=[mixture_ratio["web"], mixture_ratio["textbook"], mixture_ratio["math_reasoning"], mixture_ratio["math"], mixture_ratio["code"]],
        seed=seed,
    )

    # Take required samples, tracking per-sample token counts for split.
    # When packing, also cache the token id lists to avoid re-tokenizing later.
    full_samples: list[dict] = []
    token_counts: list[int] = []
    full_token_ids: list[list[int]] = []  # populated only when pack=True
    curr_token_count = 0
    for sample in tqdm(full_dataset, desc="Loading samples"):
        if curr_token_count >= total_tokens:
            break

        tokens = tokenizer.encode(sample["text"], add_special_tokens=False)
        full_samples.append({"text": sample["text"]})
        token_counts.append(len(tokens))
        if pack:
            full_token_ids.append(tokens)
        curr_token_count += len(tokens)

    print(f"Loaded {len(full_samples):,} samples")

    # Find split index by cumulative token count (on unshuffled order)
    train_split_idx = len(full_samples)
    cumulative = 0
    for i, n in enumerate(token_counts):
        cumulative += n
        if cumulative >= train_tokens:
            train_split_idx = i + 1
            break

    if pack:
        return _pack_and_save(
            full_token_ids=full_token_ids,
            train_split_idx=train_split_idx,
            eos_token_id=tokenizer.eos_token_id,
            seq_length=seq_length,
            seed=seed,
            output_dir=output_dir,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            mixture_ratio=mixture_ratio,
        )

    # Split first, then shuffle within each split
    full_dataset = Dataset.from_list(full_samples)
    train_dataset = full_dataset.select(range(train_split_idx)).shuffle(seed=seed)
    val_dataset   = full_dataset.select(range(train_split_idx, len(full_dataset))).shuffle(seed=seed)

    train_samples = len(train_dataset)
    val_samples   = len(val_dataset)
    print(f"\nSplitting dataset (train: {train_samples:,}, val: {val_samples:,})...")

    # Save dataset stats
    stats = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "target_train_tokens": train_tokens,
        "target_val_tokens": val_tokens,
        "seq_length": seq_length,
        "seed": seed,
        "mixture_ratio": mixture_ratio,
    }

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

    # Save dataset_info.json with relative paths (portable across machines)
    ratio_tag = "_".join(f"{k}{int(v*100)}" for k, v in sorted(mixture_ratio.items()))
    dataset_info = {
        f"mix_{ratio_tag}_train_{train_tokens/1e6:g}M": {
            "file_name": "dataset/train.arrow",
            "format": "arrow",
            "columns": {"prompt": "text"},
        },
        f"mix_{ratio_tag}_val_{val_tokens/1e6:g}M": {
            "file_name": "dataset/validation.arrow",
            "format": "arrow",
            "columns": {"prompt": "text"},
        },
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
    ratio_tag  = "_".join(f"{k}{int(v*100)}" for k, v in sorted(mixture_ratio.items()))
    train_name = f"mix_{ratio_tag}_train"
    val_name   = f"mix_{ratio_tag}_val"
    print("\nDataset names:")
    print(f"   - {train_name} (for training)")
    print(f"   - {val_name} (for validation)")
    print("\nExample training command:")
    print(f"   python scripts/prepare_mix_dataset.py --dataset {train_name} \\")
    print(f"       --dataset_dir {output_dir} --do_eval --eval_dataset {val_name}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare mix dataset for pretraining",
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
        "--dry_run",
        action="store_true",
        help="Print what would be done without actually preparing dataset",
    )

    parser.add_argument(
        "--web_dir",
        type=str,
        default="./data/fineweb",
        help="Directory containing web (fineweb) parquet files",
    )
    parser.add_argument(
        "--codes_dir",
        type=str,
        default="./data/codes",
        help="Directory containing codes jsonl.gz files",
    )
    parser.add_argument(
        "--textbook_dir",
        type=str,
        default="./data/textbook",
        help="Directory containing textbook jsonl.gz files",
    )
    parser.add_argument(
        "--math_dir",
        type=str,
        default="./data/math",
        help="Directory containing math (finemath) jsonl.gz files",
    )
    parser.add_argument(
        "--math_reasoning_dir",
        type=str,
        default="./data/math_reasoning",
        help="Directory containing math reasoning (mathl3) jsonl.gz files",
    )

    parser.add_argument(
        "--mixture_ratio",
        type=str,
        # JSON string specifying the fraction of tokens per domain. Must sum to 1.0.
        # Keys: "math", "math_reasoning", "textbook", "web"
        # Example: '{"math":0.2,"math_reasoning":0.2,"textbook":0.3,"web":0.3}'
        default='{"math": 0.05, "code": 0.1, "math_reasoning": 0.05, "textbook": 0.20, "web": 0.60}',
        help='Mixture ratio as JSON string',
    )
    parser.add_argument(
        "--weights",
        type=str,
        # Path to mix_weights.json produced by mix_rate.py (DoReMi).
        # If set, overrides --mixture_ratio with the "optimal_weights" field.
        default=None,
        help="Path to mix_weights.json from mix_rate.py (overrides --mixture_ratio)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle each domain before sampling and shuffle final dataset",
    )
    parser.add_argument(
        "--pack",
        action="store_true",
        help=(
            "Pack tokenized documents into fixed-length sequences of --seq_length tokens. "
            "Saves input_ids Arrow files instead of text Arrow files. "
            "Use scripts/pretrain_packed.py to train on the output."
        ),
    )

    args = parser.parse_args()

    
    # Read mixture weights from json file
    if args.weights:
        raw = json.loads(Path(args.weights).read_text())
        mixture_ratio = raw.get("optimal_weights", raw)
    else:
        mixture_ratio = json.loads(args.mixture_ratio)

    mix_input_dir = {
        "web":            args.web_dir,
        "code":           args.codes_dir,
        "textbook":       args.textbook_dir,
        "math":           args.math_dir,
        "math_reasoning": args.math_reasoning_dir,
    }

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
        mix_input_dir=mix_input_dir,
        mixture_ratio=mixture_ratio,
        seed=args.seed,
        tokenizer_name=args.tokenizer,
        dry_run=args.dry_run,
        pack=args.pack,
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
