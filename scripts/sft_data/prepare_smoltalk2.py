"""Prepare HuggingFaceTB/smoltalk2 (SFT subset) for the 6-token SFT mix.

Source: https://huggingface.co/datasets/HuggingFaceTB/smoltalk2

We use the SFT subset and the general-chat style subsets (avoiding the
reasoning subsets, which overlap with AM-R1 / OpenR1-Math). Each row
is already in standard `messages` format:

    {
      "messages": [
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."},
          ...
      ]
    }

We keep only single-turn or first-turn user->assistant pairs to bound
seq-len, and wrap the assistant reply in <answer>.

Usage:
    python scripts/sft_data/prepare_smoltalk2.py --output_dir /path/to/out --n 15000
    python scripts/sft_data/prepare_smoltalk2.py --output_dir /tmp/smoke --smoke 5
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
    write_jsonl,
)

SOURCE = "smoltalk2"
HF_DATASET = "HuggingFaceTB/smoltalk2"

# smoltalk2's SFT config uses subset names as split names. We target general-
# chat subsets (the `_no_think` variants avoid reasoning overlap with
# AM-R1 / OpenR1-Math).
DEFAULT_CONFIG = "SFT"
DEFAULT_SUBSETS = (
    "OpenHermes_2.5_no_think",
    "smoltalk_smollm3_everyday_conversations_no_think",
    "smoltalk_smollm3_systemchats_30k_no_think",
    "tulu_3_sft_personas_instruction_following_no_think",
)


def row_to_record(row: dict) -> dict:
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return {}

    system: str | None = None
    user: str | None = None
    assistant: str | None = None
    for m in msgs:
        role = (m.get("role") or "").lower()
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if role == "system" and system is None:
            system = content
        elif role == "user" and user is None:
            user = content
        elif role == "assistant" and assistant is None:
            assistant = content
            # stop after the first assistant reply
            break

    if user is None or not assistant:
        return {}

    wrapped = wrap_answer(assistant)
    if detect_unclosed_tokens(wrapped):
        return {}
    return make_record(source=SOURCE, user=user, assistant=wrapped, system=system)


def _interleave_subsets(config: str, subsets: list[str]):
    """Round-robin across subsets so smoke mode sees more than one source."""
    streams = []
    for name in subsets:
        try:
            streams.append((name, iter(load_dataset(HF_DATASET, config, split=name, streaming=True))))
        except Exception as e:
            print(f"[smoltalk2] subset {name!r} failed ({e}); skipping")
    if not streams:
        raise RuntimeError("no smoltalk2 subsets could be loaded")

    active = list(streams)
    while active:
        next_active = []
        for name, it in active:
            try:
                row = next(it)
            except StopIteration:
                continue
            row.setdefault("_subset", name)
            yield row
            next_active.append((name, it))
        active = next_active


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare smoltalk2 (SFT subset)")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n", type=int, default=15000)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="smoltalk2 config (e.g. 'SFT', 'Mid', 'Preference')",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(DEFAULT_SUBSETS),
        help="smoltalk2 subset names (used as split= in load_dataset)",
    )
    args = parser.parse_args()

    iterator = _interleave_subsets(args.config, args.subsets)

    if args.smoke > 0:
        print(f"[smoltalk2] SMOKE mode: first {args.smoke} rows across {args.subsets}")
        raw_rows = list(take_n(iterator, args.smoke))
    else:
        print(f"[smoltalk2] Reservoir-sampling {args.n} rows across {args.subsets}")
        raw_rows = reservoir_sample(iterator, args.n, seed=args.seed)

    records = [row_to_record(r) for r in raw_rows]
    records = [r for r in records if r]

    out_path = args.output_dir / f"{SOURCE}.jsonl"
    n_written = write_jsonl(records, out_path)
    print(f"[smoltalk2] wrote {n_written} records to {out_path}")
    if args.smoke:
        print_sample(records, k=2)


if __name__ == "__main__":
    main()
