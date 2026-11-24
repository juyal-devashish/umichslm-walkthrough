"""Prepare math datasets for GRPO training.

Outputs JSONL with a string `prompt` (pre-formatted for the 6-token schema)
and a `answer` field (gold answer for verifiable reward computation).

Supported datasets:
  gsm8k  — openai/gsm8k  (~7.5K train / 1.3K test)
  math   — lighteval/MATH (~7.5K train / 5K test; competition-level)

Usage:
    python scripts/grpo_data/prepare_grpo_math.py \\
        --dataset gsm8k \\
        --output_dir /path/to/data

    python scripts/grpo_data/prepare_grpo_math.py \\
        --dataset gsm8k \\
        --output_dir /tmp/smoke --smoke 10

    python scripts/grpo_data/prepare_grpo_math.py \\
        --dataset math \\
        --output_dir /path/to/data \\
        --n 5000
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "sft_data"))
from rewrap_utils import print_sample, reservoir_sample, take_n, write_jsonl  # noqa: E402

_SYSTEM = (
    "You are a precise mathematical reasoner. "
    "Think step-by-step inside <think>...</think>, "
    "then state your final answer inside <answer>...</answer>."
)

_PROMPT_TEMPLATE = "System: {system}\n\nProblem: {problem}\n\nSolution:"


# ---------------------------------------------------------------------------
# Per-dataset converters
# ---------------------------------------------------------------------------

def _extract_gsm8k_answer(text: str) -> str | None:
    """Pull the numeric answer from GSM8K's '#### 42' suffix."""
    m = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", text)
    return m.group(1).replace(",", "") if m else None


def _extract_math_answer(solution: str) -> str | None:
    """Pull the answer from MATH's \\boxed{...} field."""
    m = re.search(r"\\boxed\{([^}]*)\}", solution)
    return m.group(1).strip() if m else None


def _gsm8k_row(row: dict) -> dict:
    question = (row.get("question") or "").strip()
    answer = _extract_gsm8k_answer(row.get("answer") or "")
    if not question or not answer:
        return {}
    return {
        "prompt": _PROMPT_TEMPLATE.format(system=_SYSTEM, problem=question),
        "answer": answer,
        "source": "gsm8k",
    }


def _math_row(row: dict) -> dict:
    problem = (row.get("problem") or "").strip()
    answer = _extract_math_answer(row.get("solution") or "")
    if not problem or not answer:
        return {}
    return {
        "prompt": _PROMPT_TEMPLATE.format(system=_SYSTEM, problem=problem),
        "answer": answer,
        "source": "math",
    }


# (hf_path, subset, train_split, val_split, converter)
_DATASETS: dict[str, tuple] = {
    "gsm8k": ("openai/gsm8k", "main", "train", "test", _gsm8k_row),
    "math":  ("lighteval/MATH", "all", "train", "test", _math_row),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_split(
    hf_path: str,
    subset: str,
    split: str,
    n: int | None,
    smoke: int,
    seed: int,
    converter,
) -> list[dict]:
    ds = load_dataset(hf_path, subset, split=split, streaming=True)
    if smoke > 0:
        rows = list(take_n(iter(ds), smoke))
    elif n is None:
        rows = list(iter(ds))
    else:
        rows = reservoir_sample(iter(ds), n, seed=seed)
    records = [converter(r) for r in rows]
    records = [r for r in records if r]
    print(f"[{hf_path}:{split}] kept {len(records)} / {len(rows)}")
    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare math data for GRPO")
    parser.add_argument("--dataset", required=True, choices=list(_DATASETS))
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n", type=int, default=None, help="Max train samples (None = full split)")
    parser.add_argument("--smoke", type=int, default=0, help="Quick test: use first N rows only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hf_path, subset, train_split, val_split, converter = _DATASETS[args.dataset]

    train = _build_split(hf_path, subset, train_split, args.n, args.smoke, args.seed, converter)
    val   = _build_split(hf_path, subset, val_split,   None,   args.smoke, args.seed, converter)

    write_jsonl(train, args.output_dir / "train.jsonl")
    write_jsonl(val,   args.output_dir / "val.jsonl")

    if args.smoke:
        print_sample(train, k=2)

    print(f"Wrote {len(train)} train / {len(val)} val records to {args.output_dir}")


if __name__ == "__main__":
    main()
