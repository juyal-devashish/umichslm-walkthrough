"""Filter FineWeb-Edu parquets by quality score threshold.

Two-phase approach to avoid loading the full text dataset into memory:

  Phase 1 — split score map by source file, writing per-file ID lists for
             rows whose score >= threshold. (Single-process streaming pass.)

  Phase 2 — for each original parquet that has a corresponding ID list,
             stream through it and keep only matching rows.
             (Parallel across files.)

The two-phase design keeps peak memory proportional to the number of kept IDs
per file, not the full dataset size.

Usage:
    python scripts/data_cleaning/filter_by_score.py \\
        --score_map  /path/to/data \\
        --data_dir   /path/to/data \\
        --output_dir /path/to/data \\
        --threshold  3.0

    # Stricter filter
    python scripts/data_cleaning/filter_by_score.py \\
        --score_map  /path/to/data \\
        --data_dir   /path/to/data \\
        --output_dir /path/to/data \\
        --threshold  4.0 \\
        --workers 4
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Phase 1: split score map into per-file ID lists
# ---------------------------------------------------------------------------

def _split_score_map(
    score_map_path: str,
    tmp_dir: str,
    threshold: float,
) -> dict[str, str]:
    """Stream score map, write per-file ID parquets for score >= threshold.

    Returns: mapping {original_basename → ids_parquet_path}
    """
    os.makedirs(tmp_dir, exist_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    id_paths: dict[str, str] = {}
    seen: set = set()
    total = kept = 0
    start = time.time()

    pf = pq.ParquetFile(score_map_path)
    for batch in pf.iter_batches(batch_size=10_000, columns=["id", "score", "file_path"]):
        rows = pa.Table.from_batches([batch]).to_pylist()
        groups: dict[str, list] = {}
        for row in rows:
            total += 1
            sid = row["id"]
            if sid in seen:
                continue
            seen.add(sid)
            if row["score"] >= threshold:
                kept += 1
                fp = row["file_path"]
                groups.setdefault(fp, []).append(sid)

        for fp, ids in groups.items():
            base = os.path.basename(fp).replace(".parquet", "_ids.parquet")
            out = os.path.join(tmp_dir, base)
            tbl = pa.table({"id": ids})
            if fp not in writers:
                writers[fp] = pq.ParquetWriter(out, tbl.schema)
                id_paths[os.path.basename(fp)] = out
            writers[fp].write_table(tbl)

        if total % 1_000_000 == 0:
            elapsed = time.time() - start
            print(
                f"  [split] {total:,} scanned | {kept:,} kept | "
                f"{total / elapsed:,.0f} rows/s | {elapsed:.1f}s"
            )

    for w in writers.values():
        w.close()

    elapsed = time.time() - start
    print(
        f"  [split] Done: {kept:,} / {total:,} kept "
        f"across {len(writers)} files in {elapsed:.1f}s"
    )
    return id_paths


# ---------------------------------------------------------------------------
# Phase 2: filter original parquets (parallel worker)
# ---------------------------------------------------------------------------

def _filter_file(args: tuple[str, str, str]) -> tuple[str, int, int]:
    """Keep only rows whose id appears in the keep-set."""
    original_path, ids_path, output_dir = args

    keep_ids: set = set(pq.read_table(ids_path, columns=["id"]).column("id").to_pylist())
    if not keep_ids:
        return original_path, 0, 0

    out_path = os.path.join(output_dir, os.path.basename(original_path))
    pf = pq.ParquetFile(original_path)
    writer = None
    total = kept = 0

    for batch in pf.iter_batches(batch_size=1024):
        tbl = pa.Table.from_batches([batch])
        total += len(tbl)
        mask = [sid in keep_ids for sid in tbl.column("id").to_pylist()]
        filtered = tbl.filter(mask)
        if len(filtered) > 0:
            if writer is None:
                writer = pq.ParquetWriter(out_path, filtered.schema)
            writer.write_table(filtered)
            kept += len(filtered)

    if writer:
        writer.close()
    return original_path, total, kept


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter FineWeb-Edu parquets by quality score",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--score_map",  required=True, help="Score map parquet (from extract_scores.py)")
    parser.add_argument("--data_dir",   required=True, help="Original parquet directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for filtered parquets")
    parser.add_argument("--threshold",  type=float, default=3.0, help="Minimum score to keep")
    parser.add_argument("--workers",    type=int, default=min(cpu_count(), 8))
    parser.add_argument("--glob",       default="train-*.parquet")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "_tmp_id_splits")

    # Phase 1
    print(f"Phase 1: splitting score map (threshold={args.threshold}) ...")
    id_paths = _split_score_map(args.score_map, tmp_dir, args.threshold)

    # Phase 2
    print(f"\nPhase 2: filtering {len(id_paths)} parquet files ...")
    original_files = sorted(glob.glob(os.path.join(args.data_dir, args.glob)))
    work = [
        (f, id_paths[os.path.basename(f)], args.output_dir)
        for f in original_files
        if os.path.basename(f) in id_paths
    ]
    print(f"  {len(original_files)} original | {len(id_paths)} id-files | {len(work)} matched")

    start = time.time()
    total_all = kept_all = 0
    num_workers = min(args.workers, len(work))

    with Pool(num_workers) as pool:
        for path, total, kept in pool.imap_unordered(_filter_file, work):
            total_all += total
            kept_all  += kept
            elapsed = time.time() - start
            print(
                f"  {os.path.basename(path)} | {kept}/{total} kept | "
                f"running: {kept_all:,}/{total_all:,} | {elapsed:.1f}s"
            )

    # Cleanup
    for p in id_paths.values():
        os.remove(p)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    elapsed = time.time() - start
    pct = 100 * kept_all / total_all if total_all else 0
    print(f"\nDone: {kept_all:,} / {total_all:,} kept ({pct:.1f}%) in {elapsed:.1f}s")
    print(f"Output → {args.output_dir}")


if __name__ == "__main__":
    main()
