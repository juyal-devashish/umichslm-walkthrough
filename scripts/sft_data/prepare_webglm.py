"""Prepare THUDM/webglm-qa for the 6-token SFT mix.

Source: https://huggingface.co/datasets/THUDM/webglm-qa

Schema (typical):
    {
        "question": "...",
        "answer":   "Text with [1] and [2] citations ...",
        "references": ["ref1 text", "ref2 text", ...],
    }

We turn the references into a <retrieve>/<result> prelude (so the model
learns that citations follow retrieved context) and wrap all [n] citation
markers inside the answer as <cite>[n]</cite>. The full assistant message
is then wrapped in <answer>.

Usage:
    python scripts/sft_data/prepare_webglm.py --output_dir /path/to/out --n 10000
    python scripts/sft_data/prepare_webglm.py --output_dir /tmp/smoke --smoke 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from rewrap_utils import (  # noqa: E402
    detect_unclosed_tokens,
    make_record,
    print_sample,
    reservoir_sample,
    take_n,
    wrap_answer,
    wrap_citation_markers,
    write_jsonl,
)

SOURCE = "webglm_qa"
HF_DATASET = "THUDM/webglm-qa"


def row_to_record(row: dict) -> dict:
    question = row.get("question") or ""
    answer = row.get("answer") or ""
    refs = row.get("references") or []
    if not question or not answer:
        return {}

    # Build retrieved-context block. We use the question as the retrieval
    # query payload, and concatenate references as results. This teaches
    # the model the shape: <retrieve>query</retrieve><result>...</result>.
    if refs:
        refs_joined = "\n".join(f"[{i + 1}] {r}" for i, r in enumerate(refs) if r)
        preamble = f"<retrieve>{question}</retrieve><result>{refs_joined}</result>"
    else:
        preamble = ""

    # wrap every [n] marker within the answer in <cite>
    answer_with_cites = wrap_citation_markers(answer)
    assistant = preamble + wrap_answer(answer_with_cites)

    if detect_unclosed_tokens(assistant):
        return {}
    return make_record(source=SOURCE, user=question, assistant=assistant)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare THUDM/webglm-qa")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    ds = load_dataset(HF_DATASET, split=args.split, streaming=True)

    if args.smoke > 0:
        print(f"[webglm] SMOKE mode: first {args.smoke} rows")
        raw_rows = list(take_n(iter(ds), args.smoke))
    else:
        print(f"[webglm] Reservoir-sampling {args.n} rows")
        raw_rows = reservoir_sample(iter(ds), args.n, seed=args.seed)

    records = [row_to_record(r) for r in raw_rows]
    records = [r for r in records if r]

    out_path = args.output_dir / f"{SOURCE}.jsonl"
    n_written = write_jsonl(records, out_path)
    print(f"[webglm] wrote {n_written} records to {out_path}")
    if args.smoke:
        print_sample(records, k=2)


if __name__ == "__main__":
    main()
