"""Optional Claude-written change note. Runs only when ANTHROPIC_API_KEY is set.

Sends only the redacted, already-gated diff of files that are about to be
published. One request per sync; any failure returns an empty note so the
sync itself is never blocked.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from pathlib import Path

MAX_DIFF_CHARS = 60_000
PROMPT = (
    "Below is a diff of a public code walkthrough for a small language model research codebase. "
    "Write 2-4 sentences for a CHANGELOG describing what changed, in plain technical prose. "
    "Name modules and techniques. Do not mention people, institutions, clusters or file counts.\n\n"
)


def build_prompt(diff: str) -> str:
    return PROMPT + diff[:MAX_DIFF_CHARS]


def daily_note(target: Path) -> str:
    diff = subprocess.run(["git", "diff", "HEAD", "--", "src", "docs", "configs"], cwd=target,
                          capture_output=True, text=True).stdout
    if not diff.strip():
        return ""
    body = json.dumps({
        "model": os.environ.get("SUMMARY_MODEL", "claude-haiku-5-5"),
        "max_tokens": 300,
        "messages": [{"role": "user", "content": build_prompt(diff)}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body, method="POST",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["content"][0]["text"].strip()
    except Exception as e:  # never block the sync on the note
        print(f"change note skipped: {type(e).__name__}")
        return ""
