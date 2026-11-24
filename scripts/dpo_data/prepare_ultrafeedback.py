"""Prepare HuggingFaceH4/ultrafeedback_binarized for DPO.

Writes ShareGPT-style DPO records where chosen/rejected assistant responses
are wrapped in <answer>...</answer> to match the 6-token SFT format:

    {
      "conversations": [{"from": "human", "value": "<prompt>"}],
      "chosen":   {"from": "gpt", "value": "<answer>...good reply...</answer>"},
      "rejected": {"from": "gpt", "value": "<answer>...weak reply...</answer>"},
      "source": "ultrafeedback"
    }

Train comes from `train_prefs` (reservoir-sampled to --n); validation uses
the dataset's own `test_prefs` split fully.

Usage:
    python scripts/dpo_data/prepare_ultrafeedback.py \\
        --output_dir /path/to/data \\
        --n 15000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "sft_data"))
from rewrap_utils import (  # noqa: E402
    detect_unclosed_tokens,
    reservoir_sample,
    take_n,
    wrap_answer,
)

HF_DATASET = "HuggingFaceH4/ultrafeedback_binarized"
SOURCE = "ultrafeedback"


def _last_assistant(messages: list | None) -> str | None:
    if not isinstance(messages, list):
        return None
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").lower()
        content = m.get("content")
        if role == "assistant" and isinstance(content, str) and content.strip():
            return content
    return None


def row_to_record(row: dict) -> dict:
    prompt = row.get("prompt")
    chosen_text = _last_assistant(row.get("chosen"))
    rejected_text = _last_assistant(row.get("rejected"))
    if not isinstance(prompt, str) or not prompt.strip():
        return {}
    if not chosen_text or not rejected_text:
        return {}

    chosen_wrapped = wrap_answer(chosen_text)
    rejected_wrapped = wrap_answer(rejected_text)
    if detect_unclosed_tokens(chosen_wrapped) or detect_unclosed_tokens(rejected_wrapped):
        return {}

    return {
        "conversations": [{"from": "human", "value": prompt}],
        "chosen": {"from": "gpt", "value": chosen_wrapped},
        "rejected": {"from": "gpt", "value": rejected_wrapped},
        "source": SOURCE,
    }


def _write_json_array(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=None)


def _build_split(split: str, n: int | None, smoke: int, seed: int) -> list[dict]:
    ds = load_dataset(HF_DATASET, split=split, streaming=True)
    if smoke > 0:
        print(f"[ultrafeedback:{split}] SMOKE: first {smoke}")
        rows = list(take_n(iter(ds), smoke))
    elif n is None:
        print(f"[ultrafeedback:{split}] consuming full stream")
        rows = list(iter(ds))
    else:
        print(f"[ultrafeedback:{split}] reservoir-sampling {n}")
        rows = reservoir_sample(iter(ds), n, seed=seed)
    records = [row_to_record(r) for r in rows]
    records = [r for r in records if r]
    print(f"[ultrafeedback:{split}] kept {len(records)} / {len(rows)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UltraFeedback binarized for DPO")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n", type=int, default=15000, help="train target size")
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_split", type=str, default="train_prefs")
    parser.add_argument("--val_split", type=str, default="test_prefs")
    args = parser.parse_args()

    train = _build_split(args.train_split, args.n, args.smoke, args.seed)
    val = _build_split(args.val_split, None, args.smoke, args.seed)

    _write_json_array(train, args.output_dir / "train.json")
    _write_json_array(val, args.output_dir / "val.json")

    stats = {
        "train": len(train),
        "val": len(val),
        "source": SOURCE,
        "seed": args.seed,
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2)
    )
    print(f"wrote {args.output_dir}/train.json, val.json, stats.json")


if __name__ == "__main__":
    main()
