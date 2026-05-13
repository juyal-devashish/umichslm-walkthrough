"""Extract quality scores from FineWeb-Edu parquet metadata into a compact score map.

FineWeb-Edu stores per-document quality scores inside a nested `metadata` column.
This script does a metadata-only streaming pass (reads only `id` and `metadata`
columns — much faster than reading `text`) and writes a compact
`(id, score, file_path)` parquet that downstream steps use for threshold filtering.

The output is intentionally small: ~32 bytes per row, so even 100M rows fit in ~3 GB.

Usage:
    python scripts/data_cleaning/extract_scores.py \\
        --data_dir /scratch/.../fineweb-edu-dedup \\
        --output   /scratch/.../score_map.parquet

    # Limit workers (default: min(cpu_count, 8))
    python scripts/data_cleaning/extract_scores.py \\
        --data_dir /scratch/.../fineweb-edu-dedup \\
        --output   /scratch/.../score_map.parquet \\
        --workers 4
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------

def _process_file(args: tuple[str, str]) -> tuple[str, int]:
    """Extract (id, score, file_path) from one parquet file, write shard."""
    file_path, tmp_dir = args
    basename = os.path.basename(file_path).replace(".parquet", "_scores.parquet")
    out_path = os.path.join(tmp_dir, basename)

    ids: list = []
    scores: list[float] = []

    pf = pq.ParquetFile(file_path)
    for batch in pf.iter_batches(batch_size=1024, columns=["id", "metadata"]):
        rows = pa.Table.from_batches([batch]).to_pylist()
        for row in rows:
            ids.append(row["id"])
            scores.append(row["metadata"]["score"])

    out_table = pa.table({
        "id":        ids,
        "score":     np.array(scores, dtype=np.float32),
        "file_path": [file_path] * len(ids),
    })
    pq.write_table(out_table, out_path)
    return out_path, len(ids)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract FineWeb-Edu quality scores into a compact score map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",   required=True, help="Directory containing train-*.parquet files")
    parser.add_argument("--output",     required=True, help="Output score map parquet path")
    parser.add_argument("--workers",    type=int, default=min(cpu_count(), 8))
    parser.add_argument("--batch_size", type=int, default=1024, help="Parquet batch size")
    parser.add_argument("--glob",       default="train-*.parquet", help="File glob pattern")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, args.glob)))
    if not files:
        raise FileNotFoundError(f"No files matched '{args.glob}' in {args.data_dir}")
    print(f"Found {len(files)} parquet files")

    tmp_dir = str(Path(args.output).parent / "_tmp_score_shards")
    os.makedirs(tmp_dir, exist_ok=True)

    num_workers = min(args.workers, len(files))
    print(f"Using {num_workers} workers")

    start = time.time()
    total = 0
    tmp_files: list[str] = []

    with Pool(num_workers) as pool:
        for out_path, n in pool.imap_unordered(_process_file, [(f, tmp_dir) for f in files]):
            total += n
            tmp_files.append(out_path)
            elapsed = time.time() - start
            print(
                f"{total:,} rows | {len(tmp_files)}/{len(files)} files | "
                f"{total / elapsed:,.0f} rows/s | {elapsed:.1f}s"
            )

    # Merge shards
    print(f"\nMerging {len(tmp_files)} shards → {args.output} ...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for tf in sorted(tmp_files):
        t = pq.read_table(tf)
        if writer is None:
            writer = pq.ParquetWriter(args.output, t.schema)
        writer.write_table(t)
        del t
        os.remove(tf)
    if writer:
        writer.close()

    os.rmdir(tmp_dir)
    elapsed = time.time() - start
    print(f"Done: {total:,} rows in {elapsed:.1f}s ({total / elapsed:,.0f} rows/s)")
    print(f"Score map → {args.output}")


if __name__ == "__main__":
    main()
