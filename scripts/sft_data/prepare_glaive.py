"""Prepare glaive-function-calling-v2 for the 6-token SFT mix.

Source: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2

Schema (typical):
    {
        "system": "You are a helpful assistant with access to the following functions...",
        "chat": "USER: ...\\n\\n\\nASSISTANT: <functioncall> {\\"name\\":...}\\n\\n\\nFUNCTION RESPONSE: {...}\\n\\n\\nASSISTANT: ...",
    }

The conversation is a single flat string with `USER:`, `ASSISTANT:`, and
`FUNCTION RESPONSE:` delimiters. We:
  - split it into turns,
  - rename <functioncall>...</functioncall> → <retrieve>...</retrieve>,
  - wrap FUNCTION RESPONSE payloads in <result>...</result>,
  - wrap the final ASSISTANT answer in <answer>...</answer>.

Only the first user -> assistant(retrieve) -> result -> assistant(answer)
exchange is kept — multi-turn chains are truncated to keep seq len bounded.

Usage:
    python scripts/sft_data/prepare_glaive.py --output_dir /path/to/out --n 15000
    python scripts/sft_data/prepare_glaive.py --output_dir /tmp/smoke --smoke 5
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
    make_record,
    print_sample,
    reservoir_sample,
    take_n,
    wrap_answer,
    write_jsonl,
)

SOURCE = "glaive_fc"
HF_DATASET = "glaiveai/glaive-function-calling-v2"

_TURN_SPLIT = re.compile(
    r"\n*\s*(USER:|ASSISTANT:|FUNCTION RESPONSE:)\s*",
    re.IGNORECASE,
)
_FUNCTIONCALL = re.compile(
    r"<functioncall>\s*(.*?)\s*</functioncall>",
    re.DOTALL | re.IGNORECASE,
)


def parse_chat(chat: str) -> list[tuple[str, str]]:
    """Split the flat chat string into (role, content) tuples in order."""
    parts = _TURN_SPLIT.split(chat)
    # parts = [prefix, 'USER:', body, 'ASSISTANT:', body, ...]
    turns: list[tuple[str, str]] = []
    i = 1
    while i < len(parts) - 1:
        marker = parts[i].upper().rstrip(":")
        body = parts[i + 1].strip()
        if body:
            turns.append((marker, body))
        i += 2
    return turns


def rewrite_assistant(body: str) -> tuple[str, bool]:
    """Return (rewritten, is_function_call). function calls get <retrieve> wrap."""
    match = _FUNCTIONCALL.search(body)
    if match:
        payload = match.group(1).strip()
        return f"<retrieve>{payload}</retrieve>", True
    return body.strip(), False


def row_to_record(row: dict) -> dict:
    system = row.get("system") or ""
    chat = row.get("chat")
    if not chat:
        return {}

    turns = parse_chat(chat)
    if not turns:
        return {}

    # find first user turn
    user_idx = next((i for i, (r, _) in enumerate(turns) if r == "USER"), None)
    if user_idx is None:
        return {}
    user = turns[user_idx][1]

    # collect everything from user_idx+1 into the assistant payload
    tail = turns[user_idx + 1 :]
    if not tail:
        return {}

    # build assistant message by stitching retrieve/result/answer
    assistant_parts: list[str] = []
    saw_answer = False
    for role, body in tail:
        if role == "ASSISTANT":
            rewritten, is_call = rewrite_assistant(body)
            if is_call:
                assistant_parts.append(rewritten)
            else:
                # plain final answer
                assistant_parts.append(wrap_answer(rewritten))
                saw_answer = True
                break
        elif role == "FUNCTION RESPONSE":
            assistant_parts.append(f"<result>{body}</result>")
        elif role == "USER":
            # multi-turn: truncate (keep only first exchange)
            break

    if not saw_answer:
        return {}

    assistant = "".join(assistant_parts)
    if detect_unclosed_tokens(assistant):
        return {}
    return make_record(
        source=SOURCE,
        user=user,
        assistant=assistant,
        system=system if system else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare glaive-function-calling-v2")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n", type=int, default=15000)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    ds = load_dataset(HF_DATASET, split=args.split, streaming=True)

    if args.smoke > 0:
        print(f"[glaive] SMOKE mode: first {args.smoke} rows")
        raw_rows = list(take_n(iter(ds), args.smoke))
    else:
        print(f"[glaive] Reservoir-sampling {args.n} rows")
        raw_rows = reservoir_sample(iter(ds), args.n, seed=args.seed)

    records = [row_to_record(r) for r in raw_rows]
    records = [r for r in records if r]

    out_path = args.output_dir / f"{SOURCE}.jsonl"
    n_written = write_jsonl(records, out_path)
    print(f"[glaive] wrote {n_written} records to {out_path}")
    if args.smoke:
        print_sample(records, k=2)


if __name__ == "__main__":
    main()
