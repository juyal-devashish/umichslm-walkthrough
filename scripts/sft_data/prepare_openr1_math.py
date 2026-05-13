"""Prepare OpenR1-Math-220k for the 6-token SFT mix.

Source: https://huggingface.co/datasets/open-r1/OpenR1-Math-220k

Each row has:
    {
        "problem": "...",
        "solution": "...",                       # reference solution, often has \\boxed
        "generations": ["<think>...</think>\\boxed{ans} ...", ...],   # R1 traces
        "correctness_math_verify": [True, False, ...],
    }

We pick one correct generation per row, extract its final \\boxed{} as the
<answer>, and wrap the preceding reasoning in <think>.

Usage:
    python scripts/sft_data/prepare_openr1_math.py --output_dir /path/to/out --n 9000
    python scripts/sft_data/prepare_openr1_math.py --output_dir /tmp/smoke --smoke 5
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from rewrap_utils import (  # noqa: E402
    detect_unclosed_tokens,
    extract_boxed_answer,
    make_record,
    print_sample,
    reservoir_sample,
    take_n,
    wrap_answer,
    wrap_think,
    write_jsonl,
)

SOURCE = "openr1_math"
HF_DATASET = "open-r1/OpenR1-Math-220k"

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def pick_correct_generation(row: dict) -> str | None:
    gens = row.get("generations") or []
    correct = row.get("correctness_math_verify") or []
    # prefer first verified-correct trace
    for gen, ok in zip(gens, correct):
        if ok and gen:
            return gen
    # fall back to first non-empty generation
    for gen in gens:
        if gen:
            return gen
    return None


def row_to_record(row: dict) -> dict:
    problem = row.get("problem") or ""
    gen = pick_correct_generation(row)
    if not problem or not gen:
        return {}

    # try to extract <think>...</think> from the generation
    think_match = _THINK_BLOCK.search(gen)
    if think_match:
        think_text = think_match.group(1).strip()
        rest = gen.replace(think_match.group(0), "").strip()
    else:
        # some generations have the whole chain-of-thought outside tags
        think_text = ""
        rest = gen

    # pull final \boxed{} out of the remainder (or out of think_text if rest empty)
    boxed = extract_boxed_answer(rest) if rest else extract_boxed_answer(think_text)
    if boxed is None:
        # fall back to the reference solution field
        solution = row.get("solution") or ""
        boxed = extract_boxed_answer(solution)
    if boxed is None:
        return {}

    post_box_reasoning, answer = boxed

    # assemble: <think> is R1's inner trace; <answer> is the boxed result
    think_payload = think_text if think_text else post_box_reasoning
    if not think_payload.strip():
        return {}

    assistant = wrap_think(think_payload) + wrap_answer(answer)
    if detect_unclosed_tokens(assistant):
        return {}
    return make_record(source=SOURCE, user=problem, assistant=assistant)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OpenR1-Math-220k")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n", type=int, default=9000)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="HF config name; 'default' usually works",
    )
    args = parser.parse_args()

    try:
        ds = load_dataset(HF_DATASET, args.config, split=args.split, streaming=True)
    except Exception:
        # some HF datasets reject a config kwarg; retry without
        ds = load_dataset(HF_DATASET, split=args.split, streaming=True)

    if args.smoke > 0:
        print(f"[openr1_math] SMOKE mode: reading first {args.smoke} rows")
        raw_rows = list(take_n(iter(ds), args.smoke))
    else:
        print(f"[openr1_math] Reservoir-sampling {args.n} rows")
        raw_rows = reservoir_sample(iter(ds), args.n, seed=args.seed)

    records = [row_to_record(r) for r in raw_rows]
    records = [r for r in records if r]

    out_path = args.output_dir / f"{SOURCE}.jsonl"
    n_written = write_jsonl(records, out_path)
    print(f"[openr1_math] wrote {n_written} records to {out_path}")
    if args.smoke:
        print_sample(records, k=2)


if __name__ == "__main__":
    main()
