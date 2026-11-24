"""Text quality filters for pretraining data.

Applies per-document heuristic filters directly to the text column of
parquet files. Designed to catch quality issues that a numerical score
(e.g. FineWeb-Edu score) does not cover:

  - length_filter       : min/max word count
  - nonalpha_filter     : too many non-alphabetic characters (boilerplate, tables)
  - digit_filter        : too many digits (data dumps, log files)
  - line_repetition     : many duplicate lines within a doc (copy-paste artifacts)
  - bullet_repetition   : too many lines starting with the same bullet pattern
  - short_line_ratio    : docs that are mostly very short lines (menus, nav bars)

All thresholds are configurable via CLI. Filtering is applied per batch
inside each parquet file; files are processed in parallel.

Usage:
    python scripts/data_cleaning/text_quality_filters.py \\
        --input_dir  /path/to/data \\
        --output_dir /path/to/data

    # Adjusted thresholds
    python scripts/data_cleaning/text_quality_filters.py \\
        --input_dir  /path/to/data \\
        --output_dir /path/to/data \\
        --min_words 100 --max_repetition_ratio 0.2 --workers 4

    # Dry run: print stats without writing output
    python scripts/data_cleaning/text_quality_filters.py \\
        --input_dir /path/to/data \\
        --output_dir /tmp/dry \\
        --dry_run --sample 1000
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Filter configuration
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    min_words: int   = 50
    max_words: int   = 100_000
    max_nonalpha_ratio: float = 0.40   # fraction of non-space, non-alpha chars
    max_digit_ratio: float    = 0.20   # fraction of digit chars (of non-space)
    max_line_rep_ratio: float = 0.30   # fraction of duplicate lines
    max_bullet_rep_ratio: float = 0.50 # fraction of lines starting with same bullet
    max_short_line_ratio: float = 0.60 # fraction of lines with < 20 chars
    text_column: str = "text"


# ---------------------------------------------------------------------------
# Individual filters (operate on a single document string)
# ---------------------------------------------------------------------------

def _passes_length(text: str, cfg: FilterConfig) -> bool:
    n = len(text.split())
    return cfg.min_words <= n <= cfg.max_words


def _passes_nonalpha(text: str, cfg: FilterConfig) -> bool:
    chars = text.replace(" ", "").replace("\n", "")
    if not chars:
        return False
    ratio = sum(not c.isalpha() for c in chars) / len(chars)
    return ratio <= cfg.max_nonalpha_ratio


def _passes_digit(text: str, cfg: FilterConfig) -> bool:
    chars = text.replace(" ", "").replace("\n", "")
    if not chars:
        return False
    ratio = sum(c.isdigit() for c in chars) / len(chars)
    return ratio <= cfg.max_digit_ratio


def _passes_line_repetition(text: str, cfg: FilterConfig) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return True
    rep_ratio = 1.0 - len(set(lines)) / len(lines)
    return rep_ratio <= cfg.max_line_rep_ratio


def _passes_bullet_repetition(text: str, cfg: FilterConfig) -> bool:
    """Catch docs that are mostly nav menus or bullet lists with the same prefix."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 5:
        return True
    # Count lines that start with any bullet-like char: •, -, *, ·, ◦, ▪
    bullet_starts: dict[str, int] = {}
    for line in lines:
        if line and line[0] in "-*•·◦▪":
            bullet_starts[line[0]] = bullet_starts.get(line[0], 0) + 1
    if not bullet_starts:
        return True
    max_bullet = max(bullet_starts.values())
    ratio = max_bullet / len(lines)
    return ratio <= cfg.max_bullet_rep_ratio


def _passes_short_line_ratio(text: str, cfg: FilterConfig) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return True
    short = sum(1 for l in lines if len(l) < 20)
    return short / len(lines) <= cfg.max_short_line_ratio


def passes_all(text: str, cfg: FilterConfig) -> bool:
    return (
        _passes_length(text, cfg)
        and _passes_nonalpha(text, cfg)
        and _passes_digit(text, cfg)
        and _passes_line_repetition(text, cfg)
        and _passes_bullet_repetition(text, cfg)
        and _passes_short_line_ratio(text, cfg)
    )


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------

def _filter_file(args: tuple[str, str, FilterConfig, bool]) -> tuple[str, int, int]:
    file_path, output_dir, cfg, dry_run = args
    out_path = os.path.join(output_dir, os.path.basename(file_path))

    pf = pq.ParquetFile(file_path)
    writer = None
    total = kept = 0

    for batch in pf.iter_batches(batch_size=512):
        tbl = pa.Table.from_batches([batch])
        total += len(tbl)
        texts = tbl.column(cfg.text_column).to_pylist()
        mask = [passes_all(t or "", cfg) for t in texts]
        filtered = tbl.filter(mask)
        kept += len(filtered)
        if not dry_run and len(filtered) > 0:
            if writer is None:
                writer = pq.ParquetWriter(out_path, filtered.schema)
            writer.write_table(filtered)

    if writer:
        writer.close()
    return file_path, total, kept


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply text quality filters to parquet files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--glob",       default="train-*.parquet")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--workers",    type=int, default=min(cpu_count(), 8))
    # Filter thresholds
    parser.add_argument("--min_words",             type=int,   default=50)
    parser.add_argument("--max_words",             type=int,   default=100_000)
    parser.add_argument("--max_nonalpha_ratio",    type=float, default=0.40)
    parser.add_argument("--max_digit_ratio",       type=float, default=0.20)
    parser.add_argument("--max_line_rep_ratio",    type=float, default=0.30)
    parser.add_argument("--max_bullet_rep_ratio",  type=float, default=0.50)
    parser.add_argument("--max_short_line_ratio",  type=float, default=0.60)
    # Dry run: process a sample without writing output, to tune thresholds
    parser.add_argument("--dry_run", action="store_true",
                        help="Compute filter stats without writing output")
    parser.add_argument("--sample",  type=int, default=0,
                        help="If >0 with --dry_run, process only first N files")
    args = parser.parse_args()

    cfg = FilterConfig(
        min_words=args.min_words,
        max_words=args.max_words,
        max_nonalpha_ratio=args.max_nonalpha_ratio,
        max_digit_ratio=args.max_digit_ratio,
        max_line_rep_ratio=args.max_line_rep_ratio,
        max_bullet_rep_ratio=args.max_bullet_rep_ratio,
        max_short_line_ratio=args.max_short_line_ratio,
        text_column=args.text_column,
    )

    files = sorted(glob.glob(os.path.join(args.input_dir, args.glob)))
    if not files:
        raise FileNotFoundError(f"No files matched '{args.glob}' in {args.input_dir}")
    if args.dry_run and args.sample > 0:
        files = files[:args.sample]

    if not args.dry_run:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Files    : {len(files)}")
    print(f"Thresholds: min_words={cfg.min_words} max_nonalpha={cfg.max_nonalpha_ratio} "
          f"max_digit={cfg.max_digit_ratio} max_line_rep={cfg.max_line_rep_ratio} "
          f"max_bullet={cfg.max_bullet_rep_ratio} max_short_line={cfg.max_short_line_ratio}")
    if args.dry_run:
        print("[DRY RUN] no output will be written")

    work = [(f, args.output_dir, cfg, args.dry_run) for f in files]
    num_workers = min(args.workers, len(work))

    start = time.time()
    total_all = kept_all = 0

    with Pool(num_workers) as pool:
        for path, total, kept in pool.imap_unordered(_filter_file, work):
            total_all += total
            kept_all  += kept
            elapsed = time.time() - start
            pct = 100 * kept_all / total_all if total_all else 0
            print(
                f"  {os.path.basename(path)} | {kept}/{total} kept | "
                f"running: {kept_all:,}/{total_all:,} ({pct:.1f}%) | {elapsed:.1f}s"
            )

    elapsed = time.time() - start
    pct = 100 * kept_all / total_all if total_all else 0
    print(f"\nDone: {kept_all:,} / {total_all:,} kept ({pct:.1f}%) in {elapsed:.1f}s")
    if not args.dry_run:
        print(f"Output → {args.output_dir}")


if __name__ == "__main__":
    main()
