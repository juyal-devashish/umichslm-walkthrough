"""Prepare the code subset of OpenThoughts-114k for the 6-token SFT mix.

Source: https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k

Schema (typical):
    {
        "system": "...",
        "conversations": [
            {"from": "user", "value": "..."},
            {"from": "assistant",
             "value": "<|begin_of_thought|>...<|end_of_thought|>
                       <|begin_of_solution|>...```python\\n...```...<|end_of_solution|>"},
        ],
        "domain": "code" | "math" | "science" | ...,
    }

We keep only the code domain, convert thought/solution markers to
<think>/<answer>, and rewrap ``` fences to <code>.

Usage:
    python scripts/sft_data/prepare_openthoughts_code.py --output_dir /path/to/out --n 10000
    python scripts/sft_data/prepare_openthoughts_code.py --output_dir /tmp/smoke --smoke 5
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from rewrap_utils import (  # noqa: E402
    convert_code_fences,
    detect_unclosed_tokens,
    make_record,
    print_sample,
    reservoir_sample,
    take_n,
    wrap_answer,
    wrap_think,
    write_jsonl,
)

SOURCE = "openthoughts_code"
HF_DATASET = "open-thoughts/OpenThoughts-114k"

_THOUGHT = re.compile(
    r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL
)
_SOLUTION = re.compile(
    r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL
)

# rows tagged as code; the dataset uses a few different label keys
CODE_DOMAIN_KEYS = ("domain", "category", "task_type", "type")
CODE_LABELS = {"code", "coding", "programming", "algorithm"}


def is_code_row(row: dict) -> bool:
    for key in CODE_DOMAIN_KEYS:
        val = row.get(key)
        if isinstance(val, str) and val.lower() in CODE_LABELS:
            return True
    # fall back to content heuristic: assistant text mentions ```python or def
    conv = row.get("conversations") or []
    for m in conv:
        content = m.get("value") if isinstance(m, dict) else None
        if isinstance(content, str) and ("```python" in content or "```cpp" in content or "```java" in content):
            return True
    return False


def extract_user_assistant(row: dict) -> tuple[str, str] | None:
    conv = row.get("conversations") or []
    user = None
    assistant = None
    for m in conv:
        role = (m.get("from") or m.get("role") or "").lower()
        content = m.get("value") or m.get("content")
        if not content:
            continue
        if role in ("user", "human"):
            user = content
        elif role in ("assistant", "gpt", "ai"):
            assistant = content
    if user is None or assistant is None:
        return None
    return user, assistant


def normalize_assistant(raw: str) -> str:
    thought_match = _THOUGHT.search(raw)
    solution_match = _SOLUTION.search(raw)
    if thought_match:
        think_text = thought_match.group(1).strip()
    else:
        think_text = ""
    if solution_match:
        solution_text = solution_match.group(1).strip()
    else:
        # no <|begin_of_solution|> markers — treat whole text as solution
        solution_text = raw.strip()

    solution_text = convert_code_fences(solution_text)
    if not solution_text:
        return ""
    assembled = ""
    if think_text:
        assembled += wrap_think(think_text)
    assembled += wrap_answer(solution_text)
    return assembled


def row_to_record(row: dict) -> dict:
    if not is_code_row(row):
        return {}
    turns = extract_user_assistant(row)
    if turns is None:
        return {}
    user, assistant_raw = turns
    assistant = normalize_assistant(assistant_raw)
    if not assistant or detect_unclosed_tokens(assistant):
        return {}
    system = row.get("system") if isinstance(row.get("system"), str) else None
    return make_record(source=SOURCE, user=user, assistant=assistant, system=system)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OpenThoughts-114k (code)")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--no_domain_filter",
        action="store_true",
        help="disable the code-only filter (useful if dataset lacks domain labels)",
    )
    args = parser.parse_args()

    ds = load_dataset(HF_DATASET, split=args.split, streaming=True)

    def _iter():
        for row in ds:
            if args.no_domain_filter or is_code_row(row):
                yield row

    if args.smoke > 0:
        print(f"[openthoughts_code] SMOKE mode: first {args.smoke} code rows")
        raw_rows = list(take_n(_iter(), args.smoke))
    else:
        print(f"[openthoughts_code] Reservoir-sampling {args.n} code rows")
        raw_rows = reservoir_sample(_iter(), args.n, seed=args.seed)

    records = [row_to_record(r) for r in raw_rows]
    records = [r for r in records if r]

    out_path = args.output_dir / f"{SOURCE}.jsonl"
    n_written = write_jsonl(records, out_path)
    print(f"[openthoughts_code] wrote {n_written} records to {out_path}")
    if args.smoke:
        print_sample(records, k=2)


if __name__ == "__main__":
    main()
