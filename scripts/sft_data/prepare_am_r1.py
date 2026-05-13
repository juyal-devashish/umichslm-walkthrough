"""Prepare AM-DeepSeek-R1-Distilled-1.4M for the 6-token SFT mix.

Source: https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M

Assistant responses already follow:
    <think>reasoning</think>
    <answer>final answer</answer>

...but the exact field name and separators vary by release. This preparer
defensively extracts think/answer blocks or reconstructs them from raw text.

Usage:
    python scripts/sft_data/prepare_am_r1.py --output_dir /path/to/out --n 21000
    python scripts/sft_data/prepare_am_r1.py --output_dir /tmp/smoke --smoke 5
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from datasets import load_dataset

# make rewrap_utils importable when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from rewrap_utils import (  # noqa: E402
    detect_unclosed_tokens,
    make_record,
    print_sample,
    reservoir_sample,
    take_n,
    wrap_answer,
    wrap_think,
    write_jsonl,
)

SOURCE = "am_r1_distilled"
HF_DATASET = "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"

# possible field names the dataset might use for the conversation
CANDIDATE_MSG_KEYS = ("messages", "conversations", "conversation")
CANDIDATE_ROLE_KEYS = ("role", "from")
CANDIDATE_CONTENT_KEYS = ("content", "value")

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_turns(row: dict) -> tuple[str | None, str, str] | None:
    """Extract (system, user, assistant) from a row. Returns None if unparseable."""
    msgs = None
    for key in CANDIDATE_MSG_KEYS:
        if key in row and isinstance(row[key], list):
            msgs = row[key]
            break
    if not msgs:
        return None

    system: str | None = None
    user: str | None = None
    assistant: str | None = None
    for m in msgs:
        role = None
        content = None
        for rk in CANDIDATE_ROLE_KEYS:
            if rk in m:
                role = m[rk]
                break
        for ck in CANDIDATE_CONTENT_KEYS:
            if ck in m:
                content = m[ck]
                break
        if role is None or content is None:
            continue
        role = role.lower().strip()
        if role in ("system",):
            system = content
        elif role in ("user", "human"):
            user = content
        elif role in ("assistant", "gpt", "ai"):
            assistant = content
    if user is None or assistant is None:
        return None
    return system, user, assistant


def normalize_assistant(assistant: str) -> str:
    """Ensure assistant text has exactly one <think>...</think><answer>...</answer>.

    Strategy:
      - If both blocks present: keep as-is.
      - If only <think>...</think> present: wrap remainder in <answer>.
      - If neither present: treat whole thing as answer (no think).
    """
    think_match = _THINK_BLOCK.search(assistant)
    answer_match = _ANSWER_BLOCK.search(assistant)
    if think_match and answer_match:
        return assistant.strip()
    if think_match and not answer_match:
        think_text = think_match.group(0)
        rest = assistant.replace(think_text, "").strip()
        if not rest:
            return ""  # no final answer → drop
        return f"{think_text}\n{wrap_answer(rest)}"
    # no think block at all — use whole text as answer, no reasoning
    return wrap_answer(assistant)


def row_to_record(row: dict) -> dict:
    turns = extract_turns(row)
    if turns is None:
        return {}
    system, user, assistant_raw = turns
    assistant = normalize_assistant(assistant_raw)
    if not assistant or detect_unclosed_tokens(assistant):
        return {}
    return make_record(source=SOURCE, user=user, assistant=assistant, system=system)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AM-DeepSeek-R1-Distilled-1.4M")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n", type=int, default=21000, help="target samples (full mode)")
    parser.add_argument("--smoke", type=int, default=0, help="if >0, stream only N rows for smoke test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--config",
        type=str,
        default="am_0.5M",
        help="AM-R1 config: 'am_0.5M' (default, zstd), 'am_0.9M', 'am_0.9M_sample_1k'",
    )
    args = parser.parse_args()

    streaming = True  # always stream — reservoir-sample locally, keeps cache tiny
    print(f"[am_r1] loading config={args.config!r} split={args.split!r}")
    ds = load_dataset(HF_DATASET, args.config, split=args.split, streaming=streaming)

    if args.smoke > 0:
        print(f"[am_r1] SMOKE mode: reading first {args.smoke} rows")
        raw_rows = list(take_n(iter(ds), args.smoke))
    else:
        print(f"[am_r1] Reservoir-sampling {args.n} rows (this streams the whole dataset once)")
        raw_rows = reservoir_sample(iter(ds), args.n, seed=args.seed)

    records = [row_to_record(r) for r in raw_rows]
    records = [r for r in records if r]

    out_path = args.output_dir / f"{SOURCE}.jsonl"
    n_written = write_jsonl(records, out_path)
    print(f"[am_r1] wrote {n_written} records to {out_path}")
    if args.smoke:
        print_sample(records, k=2)


if __name__ == "__main__":
    main()
