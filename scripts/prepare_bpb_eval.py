"""Build the frozen BPB evaluation set (v1). One-off data prep script.

Why this script exists:
    BPB (bits-per-byte) is the only loss metric that is comparable across
    different tokenizers. To make BPB numbers comparable across runs /
    architectures / weeks / papers, the *text* being scored must be
    byte-for-byte identical. That requires freezing a fixed set of raw
    text files with a manifest of sha256 digests.

What this produces under --output_dir (default `data/eval/bpb_v1/`):
    web.txt      — ~500 KB from fineweb-edu
    code.txt     — ~500 KB from the-stack-smol (python)
    math.txt     — ~500 KB from open-web-math
    MANIFEST.json — {name, bytes, sha256} for each file + version tag

Eval set size is small (~1.5 MB raw) on purpose:
    - Fits in the repo (or at worst an LFS entry) so BPB is reproducible
      without re-downloading from HF.
    - Scoring 1.5 MB at max_length=1024, batch_size=4 takes <1 minute even
      on a single CPU core — we want BPB cheap enough to run every
      milestone checkpoint.

Usage (run once, then commit the output):
    pip install datasets
    python scripts/prepare_bpb_eval.py
    git add data/eval/bpb_v1/
    git commit -m "Add BPB v1 eval set"

Re-runs are deterministic given the same --seed and --target_bytes, but you
should NOT re-run this after the v1 set is committed — that would break
comparability of existing BPB numbers. If you need different text, bump to
v2 by passing --output_dir data/eval/bpb_v2/.
"""

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# (name, HF dataset, config, split, text column) — streamed, so no full download.
# Chosen to mirror the domain mix in prepare_mix_dataset.py as closely as possible
# while staying on permissive / public sources.
_SOURCES = [
    ("web",  "HuggingFaceFW/fineweb-edu",   "sample-10BT", "train", "text"),
    ("code", "codeparrot/codeparrot-clean", None,          "train", "content"),
    ("math", "open-web-math/open-web-math", None,          "train", "text"),
]


def _collect_bytes(stream, text_col: str, target_bytes: int, max_docs: int = 5000) -> str:
    """Pull from a streaming HF dataset until we have >= target_bytes of text."""
    out = []
    total = 0
    first_keys = None
    for i, row in enumerate(stream):
        if i == 0:
            first_keys = list(row.keys())
        if i >= max_docs:
            break
        t = row.get(text_col)
        if not isinstance(t, str) or not t.strip():
            continue
        out.append(t.strip())
        # Two newlines between docs → clear boundary without being tokenizer-specific.
        total += len(t.encode("utf-8")) + 2
        if total >= target_bytes:
            break
    if total == 0 and first_keys is not None:
        logger.error(
            "No text collected. First row had columns: %s (expected '%s'). "
            "Schema may have changed — update _SOURCES in prepare_bpb_eval.py.",
            first_keys, text_col,
        )
    return "\n\n".join(out)


def main():
    parser = argparse.ArgumentParser(
        description="Build the frozen BPB v1 eval set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", type=str, default="data/eval/bpb_v1")
    parser.add_argument(
        "--target_bytes", type=int, default=500_000,
        help="Target bytes of raw UTF-8 text per source (default ~500 KB).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--version_tag", type=str, default="v1",
        help="Written into MANIFEST.json as the eval_set_version. Bump if you "
             "intentionally change the text — never mutate an existing version.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    random.seed(args.seed)

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("prepare_bpb_eval.py requires `datasets`. Try: pip install datasets")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"eval_set_version": args.version_tag, "files": []}

    for name, repo, config, split, text_col in _SOURCES:
        logger.info("Streaming %s (%s / %s) ...", name, repo, config)
        ds = load_dataset(repo, config, split=split, streaming=True)
        text = _collect_bytes(ds, text_col, args.target_bytes)
        if len(text.encode("utf-8")) < args.target_bytes // 2:
            raise RuntimeError(
                f"Only got {len(text)} bytes for {name}; source may have changed schema."
            )

        path = out_dir / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        manifest["files"].append({
            "name": name, "path": path.name, "bytes": size, "sha256": sha,
            "source": {"repo": repo, "config": config, "split": split, "text_col": text_col},
        })
        logger.info("  wrote %s (%d bytes, sha256=%s...)", path, size, sha[:12])

    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote manifest: %s", manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
