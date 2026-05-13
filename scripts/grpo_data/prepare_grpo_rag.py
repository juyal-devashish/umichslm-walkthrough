"""Prepare TriviaQA (rc subset) for RAG-style GRPO training.

TriviaQA `rc` provides:
  - question:        str
  - answer:          dict with `value` (str) and `normalized_aliases` (list[str])
  - search_results:  dict with `description` (list[str]), `title` (list[str])

The prompt embeds the top-K retrieved passages so the model can learn to:
  1. Echo the retrieval query in <retrieve>...</retrieve>
  2. Reproduce/summarise context in <result>...</result>
  3. Answer in <answer>...</answer> with <cite>[n]</cite> markers

Three columns are written per record so reward functions can access them
as kwargs via GRPOTrainer:
  - prompt    : formatted prompt string
  - answer    : primary gold answer (normalized_value) for logging
  - aliases   : JSON-encoded list of all normalized_aliases for rag_accuracy_reward
  - context   : raw joined passage text for grounding_reward

Usage:
    python scripts/grpo_data/prepare_grpo_rag.py \\
        --output_dir /scratch/.../grpo_data/v1/triviaqa
    python scripts/grpo_data/prepare_grpo_rag.py \\
        --output_dir /tmp/smoke --smoke 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "sft_data"))
from rewrap_utils import print_sample, reservoir_sample, take_n, write_jsonl  # noqa: E402

HF_DATASET = "trivia_qa"
HF_SUBSET  = "rc"
MAX_PASSAGES = 5   # passages included in the prompt; keep prompt length manageable

_SYSTEM = (
    "You are a precise question-answering assistant. "
    "Think step-by-step inside <think>...</think>, "
    "then show the retrieved context inside "
    "<retrieve>query</retrieve><result>passages</result>, "
    "and finally answer inside <answer>text</answer> "
    "using <cite>[n]</cite> to reference the numbered passages."
)

_PROMPT_TEMPLATE = (
    "System: {system}\n\n"
    "Retrieved context:\n{context_block}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_passages(search_results: dict | None, k: int) -> list[str]:
    """Return up to k passage strings from the search_results dict."""
    if not isinstance(search_results, dict):
        return []
    descriptions = search_results.get("description") or []
    titles       = search_results.get("title") or []
    passages: list[str] = []
    for i, desc in enumerate(descriptions):
        if len(passages) >= k:
            break
        if not isinstance(desc, str) or not desc.strip():
            continue
        title = titles[i] if i < len(titles) and isinstance(titles[i], str) else ""
        entry = f"{title.strip()}: {desc.strip()}" if title.strip() else desc.strip()
        passages.append(entry)
    return passages


def _build_context_block(passages: list[str]) -> str:
    """Format passages as a numbered list for the prompt."""
    return "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))


def row_to_record(row: dict) -> dict | None:
    question = (row.get("question") or "").strip()
    if not question:
        return None

    answer_field = row.get("answer") or {}
    gold = (answer_field.get("normalized_value") or "").strip().lower()
    aliases: list[str] = [
        a.lower() for a in (answer_field.get("normalized_aliases") or [])
        if isinstance(a, str) and a.strip()
    ]
    if not gold and not aliases:
        return None
    if gold and gold not in aliases:
        aliases = [gold] + aliases

    passages = _extract_passages(row.get("search_results"), MAX_PASSAGES)
    if not passages:
        return None

    context_block = _build_context_block(passages)
    context_raw   = "\n".join(passages)  # raw text for grounding_reward

    prompt = _PROMPT_TEMPLATE.format(
        system=_SYSTEM,
        context_block=context_block,
        question=question,
    )

    return {
        "prompt":   prompt,
        "answer":   gold,                          # primary gold; used for display
        "aliases":  json.dumps(aliases),           # all acceptable answers (JSON string)
        "context":  context_raw,                   # retrieved text for grounding_reward
        "source":   "triviaqa",
    }


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------

def _build_split(split: str, n: int | None, smoke: int, seed: int) -> list[dict]:
    print(f"[triviaqa:{split}] loading {HF_DATASET}/{HF_SUBSET}")
    ds = load_dataset(HF_DATASET, HF_SUBSET, split=split, streaming=True,
                      trust_remote_code=True)

    if smoke > 0:
        raw = list(take_n(iter(ds), smoke))
    elif n is None:
        raw = list(iter(ds))
    else:
        raw = reservoir_sample(iter(ds), n, seed=seed)

    records = [row_to_record(r) for r in raw]
    records = [r for r in records if r]
    print(f"[triviaqa:{split}] kept {len(records)} / {len(raw)}")
    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TriviaQA RC for GRPO")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n",     type=int, default=None,
                        help="max train samples (None = full split)")
    parser.add_argument("--smoke", type=int, default=0,
                        help="quick test: first N rows only")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--max_passages", type=int, default=MAX_PASSAGES,
                        help=f"passages per prompt (default {MAX_PASSAGES})")
    args = parser.parse_args()

    global MAX_PASSAGES
    MAX_PASSAGES = args.max_passages

    train = _build_split("train",      args.n,    args.smoke, args.seed)
    val   = _build_split("validation", None,      args.smoke, args.seed)

    write_jsonl(train, args.output_dir / "train.jsonl")
    write_jsonl(val,   args.output_dir / "val.jsonl")

    if args.smoke:
        print_sample(
            # rewrap for print_sample which expects messages-style records
            [{"messages": [{"role": "user", "content": r["prompt"][:300]},
                           {"role": "assistant", "content": r["answer"]}],
              "source": r["source"], "tokens_used": []}
             for r in train[:2]],
            k=2,
        )

    print(f"Wrote {len(train)} train / {len(val)} val to {args.output_dir}")


if __name__ == "__main__":
    main()
