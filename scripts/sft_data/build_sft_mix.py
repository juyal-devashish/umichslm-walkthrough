"""Aggregate per-source .jsonl files into a single 80K SFT mix with val split.

Reads the 6 prepared files (produced by the `prepare_*.py` scripts) from
`--input_dir`, concatenates them, shuffles with a fixed seed, and writes:

    {output_dir}/train.json
    {output_dir}/val.json        (3% holdout, stratified by source)
    {output_dir}/stats.json      (per-source counts + token coverage)

The output is HF-Datasets-compatible JSON (a list of records) so LLaMA
Factory's `file_name:` entries can load it directly.

Usage:
    python scripts/sft_data/build_sft_mix.py \\
        --input_dir /path/to/data \\
        --output_dir /path/to/data
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rewrap_utils import read_jsonl  # noqa: E402

EXPECTED_SOURCES = (
    "am_r1_distilled",
    "openr1_math",
    "openthoughts_code",
    "glaive_fc",
    "webglm_qa",
    "smoltalk2",
)


def stratified_split(
    records: list[dict], val_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split records into train/val keeping source proportions."""
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_source[r.get("source", "_unknown")].append(r)

    train: list[dict] = []
    val: list[dict] = []
    for source, rows in by_source.items():
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * val_frac))
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def compute_stats(records: list[dict]) -> dict:
    by_source = Counter(r.get("source", "_unknown") for r in records)
    token_coverage: dict[str, int] = Counter()
    for r in records:
        for t in r.get("tokens_used", []):
            token_coverage[t] += 1
    return {
        "total": len(records),
        "by_source": dict(by_source),
        "token_coverage": dict(token_coverage),
    }


def write_json_array(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SFT mix")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--val_frac", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_records: list[dict] = []
    missing: list[str] = []
    for source in EXPECTED_SOURCES:
        path = args.input_dir / f"{source}.jsonl"
        if not path.exists():
            missing.append(source)
            continue
        source_records = list(read_jsonl(path))
        print(f"  {source}: {len(source_records)} records")
        all_records.extend(source_records)

    if missing:
        print(f"WARNING: missing sources (skipped): {missing}")

    print(f"Total pre-split: {len(all_records)}")
    train, val = stratified_split(all_records, args.val_frac, args.seed)
    print(f"Train: {len(train)}  Val: {len(val)}")

    write_json_array(train, args.output_dir / "train.json")
    write_json_array(val, args.output_dir / "val.json")

    stats = {
        "train": compute_stats(train),
        "val": compute_stats(val),
        "val_frac": args.val_frac,
        "seed": args.seed,
        "missing_sources": missing,
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2)
    )
    print(f"Wrote {args.output_dir}/train.json, val.json, stats.json")


if __name__ == "__main__":
    main()
