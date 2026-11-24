"""Prepare openai/webgpt_comparisons for RAG-style DPO.

WebGPT provides two answers per question with human preference scores.
We wrap both in the 6-token RAG schema (<retrieve>/<result>/<answer>/<cite>)
so DPO trains the model to *prefer* better-cited, more grounded responses
over weaker ones — not just grounded vs. ungrounded.

Dataset access: the standard HF loader is blocked by a legacy script
restriction; we use the parquet export instead (validated by
scripts/dpo_data/evaluate_dpo_suitability.py).

Output format (ShareGPT DPO, matches prepare_ultrafeedback.py):
    {
      "conversations": [{"from": "human", "value": "<question>"}],
      "chosen":   {"from": "gpt", "value": "<retrieve>...<result>...<answer>...<cite>..."},
      "rejected": {"from": "gpt", "value": "<retrieve>...<result>...<answer>...<cite>..."},
      "source": "webgpt"
    }

Usage:
    python scripts/dpo_data/prepare_webgpt.py --output_dir /path/to/data
    python scripts/dpo_data/prepare_webgpt.py --output_dir /tmp/smoke --smoke 10
    python scripts/dpo_data/prepare_webgpt.py --output_dir /tmp/probe --probe
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "sft_data"))
from rewrap_utils import (  # noqa: E402
    detect_unclosed_tokens,
    reservoir_sample,
    take_n,
    wrap_answer,
    wrap_citation_markers,
)

SOURCE = "webgpt"
PARQUET_PATH = (
    "hf://datasets/openai/webgpt_comparisons"
    "@refs/convert/parquet/default/train/0000.parquet"
)

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_answer_field(field) -> tuple[str, list[str]]:
    """Return (answer_text, list_of_reference_strings) from an answer_0/1 field.

    Handles both nested-dict (original schema) and flat-string (degenerate parquet)
    representations defensively.
    """
    if isinstance(field, dict):
        text = (field.get("answer") or "").strip()
        context = field.get("context") or []
        refs: list[str] = []
        for item in context:
            if not isinstance(item, dict):
                continue
            extract = (item.get("extract") or item.get("description") or "").strip()
            title = (item.get("title") or "").strip()
            if extract:
                entry = f"{title}: {extract}" if title else extract
                refs.append(entry)
        return text, refs
    if isinstance(field, str):
        return field.strip(), []
    return "", []


def _format_rag_response(text: str, refs: list[str], query: str) -> str:
    """Wrap an answer as a full RAG response using the 6-token schema.

    Structure:
        <retrieve>{query}</retrieve>
        <result>[1] ref1 \\n [2] ref2 \\n ...</result>
        <answer>text with <cite>[n]</cite> markers</answer>
    """
    if refs:
        refs_block = "\n".join(f"[{i + 1}] {r}" for i, r in enumerate(refs))
        preamble = f"<retrieve>{query}</retrieve><result>{refs_block}</result>"
    else:
        preamble = ""
    answer_with_cites = wrap_citation_markers(text)
    return preamble + wrap_answer(answer_with_cites)


# ---------------------------------------------------------------------------
# Row converter
# ---------------------------------------------------------------------------

def row_to_record(row: dict) -> dict:
    question = row.get("question") or {}
    if isinstance(question, dict):
        query = question.get("full_text", "").strip()
    else:
        query = str(question).strip()
    if not query:
        return {}

    s0 = float(row.get("score_0") or 0.0)
    s1 = float(row.get("score_1") or 0.0)
    if s0 == s1:
        return {}  # exclude ties — no preference signal

    chosen_field, rejected_field = (
        (row.get("answer_0"), row.get("answer_1")) if s0 > s1
        else (row.get("answer_1"), row.get("answer_0"))
    )

    chosen_text, chosen_refs = _extract_answer_field(chosen_field)
    rejected_text, rejected_refs = _extract_answer_field(rejected_field)
    if not chosen_text or not rejected_text:
        return {}

    chosen_val = _format_rag_response(chosen_text, chosen_refs, query)
    rejected_val = _format_rag_response(rejected_text, rejected_refs, query)

    if detect_unclosed_tokens(chosen_val) or detect_unclosed_tokens(rejected_val):
        return {}

    return {
        "conversations": [{"from": "human", "value": query}],
        "chosen":   {"from": "gpt", "value": chosen_val},
        "rejected": {"from": "gpt", "value": rejected_val},
        "source": SOURCE,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_json_array(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=None)


def _train_val_split(
    records: list[dict], val_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare WebGPT comparisons for RAG-style DPO"
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n",       type=int,   default=15000, help="train target size")
    parser.add_argument("--val_frac",type=float, default=0.05,  help="val holdout fraction")
    parser.add_argument("--smoke",   type=int,   default=0,     help="quick test: first N rows")
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument(
        "--probe", action="store_true",
        help="Print first-row schema and exit — use to verify parquet field names",
    )
    args = parser.parse_args()

    print(f"[webgpt] loading from parquet: {PARQUET_PATH}")
    ds = load_dataset(
        "parquet",
        data_files={"train": PARQUET_PATH},
        split="train",
        streaming=True,
    )

    if args.probe:
        first = next(iter(ds))
        print("=== first-row keys ===")
        for k, v in first.items():
            print(f"  {k!r}: {type(v).__name__} = {str(v)[:120]!r}")
        return

    if args.smoke > 0:
        print(f"[webgpt] SMOKE: first {args.smoke} rows")
        raw_rows = list(take_n(iter(ds), args.smoke))
    else:
        print(f"[webgpt] reservoir-sampling up to {args.n} rows")
        raw_rows = reservoir_sample(iter(ds), args.n, seed=args.seed)

    records = [row_to_record(r) for r in raw_rows]
    records = [r for r in records if r]
    print(f"[webgpt] kept {len(records)} / {len(raw_rows)} rows after filtering")

    if args.smoke:
        for rec in records[:2]:
            print("--- sample ---")
            print("Q:", rec["conversations"][0]["value"][:120])
            print("CHOSEN:",   rec["chosen"]["value"][:200])
            print("REJECTED:", rec["rejected"]["value"][:200])
        return

    train, val = _train_val_split(records, args.val_frac, args.seed)
    _write_json_array(train, args.output_dir / "train.json")
    _write_json_array(val,   args.output_dir / "val.json")

    stats = {"train": len(train), "val": len(val), "source": SOURCE, "seed": args.seed}
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(f"[webgpt] wrote {len(train)} train / {len(val)} val → {args.output_dir}")


if __name__ == "__main__":
    main()
