"""Shared utilities for rewrapping SFT data into the 6-token schema.

Schema (locked Week 1):
    <think>/</think>, <answer>/</answer>, <code>/</code>,
    <retrieve>/</retrieve>, <result>/</result>, <cite>/</cite>

All preparers output records in the same ShareGPT-style schema:
    {
        "messages": [
            {"role": "system", "content": "..."},   # optional
            {"role": "user",   "content": "..."},
            {"role": "assistant", "content": "<think>...</think><answer>...</answer>"}
        ],
        "source":      "am_r1_distilled",       # provenance tag
        "tokens_used": ["<think>", "<answer>"]  # which schema tokens appear
    }
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA_TOKENS: tuple[str, ...] = (
    "<think>", "</think>",
    "<answer>", "</answer>",
    "<code>", "</code>",
    "<retrieve>", "</retrieve>",
    "<result>", "</result>",
    "<cite>", "</cite>",
)

OPEN_CLOSE_PAIRS: tuple[tuple[str, str], ...] = (
    ("<think>",   "</think>"),
    ("<answer>",  "</answer>"),
    ("<code>",    "</code>"),
    ("<retrieve>", "</retrieve>"),
    ("<result>",  "</result>"),
    ("<cite>",    "</cite>"),
)


# ---------------------------------------------------------------------------
# Wrapping helpers
# ---------------------------------------------------------------------------

def wrap_think(text: str) -> str:
    """Wrap a reasoning trace in <think>...</think>."""
    stripped = text.strip()
    if stripped.startswith("<think>") and stripped.endswith("</think>"):
        return stripped
    return f"<think>{stripped}</think>"


def wrap_answer(text: str) -> str:
    """Wrap a final answer in <answer>...</answer>."""
    stripped = text.strip()
    if stripped.startswith("<answer>") and stripped.endswith("</answer>"):
        return stripped
    return f"<answer>{stripped}</answer>"


def wrap_cite(marker: str) -> str:
    """Wrap a citation marker like '[1]' or '1' in <cite>...</cite>."""
    return f"<cite>{marker}</cite>"


_CODE_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n?(.*?)```", re.DOTALL)


def convert_code_fences(text: str) -> str:
    """Replace ```lang\\n...\\n``` fences with <code>...</code>."""
    def _sub(match: re.Match[str]) -> str:
        body = match.group(1).strip("\n")
        return f"<code>{body}</code>"
    return _CODE_FENCE.sub(_sub, text)


_CITATION_MARKER = re.compile(r"\[(\d+)\]")


def wrap_citation_markers(text: str) -> str:
    """Replace bare [n] citations with <cite>[n]</cite>, skipping existing."""
    def _sub(match: re.Match[str]) -> str:
        full = match.group(0)
        # skip if already inside a <cite> tag (best-effort)
        span_start = match.start()
        preceding = text[max(0, span_start - 8) : span_start]
        if preceding.endswith("<cite>"):
            return full
        return f"<cite>{full}</cite>"
    return _CITATION_MARKER.sub(_sub, text)


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def extract_boxed_answer(text: str) -> tuple[str, str] | None:
    """Split 'reasoning ... \\boxed{x}' into (reasoning_without_box, x).

    Returns None if no \\boxed{...} found. Nested braces are not handled —
    OpenR1-Math uses simple boxed payloads for almost all rows.
    """
    match = _BOXED.search(text)
    if match is None:
        return None
    answer = match.group(1).strip()
    before = text[: match.start()].rstrip()
    after = text[match.end() :].rstrip()
    reasoning = (before + " " + after).strip()
    return reasoning, answer


# ---------------------------------------------------------------------------
# Closure validation
# ---------------------------------------------------------------------------

def detect_unclosed_tokens(text: str) -> list[str]:
    """Return a list of schema tokens whose open/close counts don't match."""
    problems: list[str] = []
    for open_tag, close_tag in OPEN_CLOSE_PAIRS:
        if text.count(open_tag) != text.count(close_tag):
            problems.append(open_tag)
    return problems


def detected_tokens(text: str) -> list[str]:
    """Return the set of schema open-tags that appear in the text."""
    found: list[str] = []
    for open_tag, _close in OPEN_CLOSE_PAIRS:
        if open_tag in text:
            found.append(open_tag)
    return found


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def make_record(
    *,
    source: str,
    user: str,
    assistant: str,
    system: str | None = None,
) -> dict:
    """Build a standard record. Drops records with unclosed schema tokens."""
    if detect_unclosed_tokens(assistant):
        return {}
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": user.strip()})
    messages.append({"role": "assistant", "content": assistant.strip()})
    return {
        "messages": messages,
        "source": source,
        "tokens_used": detected_tokens(assistant),
    }


# ---------------------------------------------------------------------------
# Streaming sampling
# ---------------------------------------------------------------------------

def take_n(stream: Iterable[dict], n: int) -> Iterator[dict]:
    """Yield the first n items from a streaming HF dataset."""
    for i, row in enumerate(stream):
        if i >= n:
            return
        yield row


def reservoir_sample(stream: Iterable[dict], n: int, seed: int = 42) -> list[dict]:
    """Reservoir-sample n rows from a stream of unknown length."""
    rng = random.Random(seed)
    reservoir: list[dict] = []
    for i, row in enumerate(stream):
        if i < n:
            reservoir.append(row)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = row
    return reservoir


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_jsonl(records: Iterable[dict], path: Path) -> int:
    """Write records to a .jsonl file. Skips falsy (dropped) records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            if not rec:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Smoke-mode preview
# ---------------------------------------------------------------------------

def print_sample(records: list[dict], k: int = 2) -> None:
    """Print k records for eyeballing."""
    for i, rec in enumerate(records[:k]):
        print(f"--- sample {i} (source={rec.get('source')}) ---")
        print(f"tokens_used: {rec.get('tokens_used')}")
        for msg in rec["messages"]:
            content = msg["content"]
            if len(content) > 500:
                content = content[:500] + f"...[+{len(msg['content']) - 500} chars]"
            print(f"[{msg['role']}] {content}")
        print()
