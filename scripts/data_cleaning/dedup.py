"""Near-duplicate removal using MinHash LSH.

Uses datasketch's MinHashLSH to identify and remove near-duplicate documents
based on word-level Jaccard similarity.

Two-pass algorithm:
  Pass 1 — stream all parquet files, compute MinHash signature per document,
            query LSH index (built incrementally), and record which IDs to drop.
            (Single process — LSH index must be shared.)
  Pass 2 — stream files again, write only non-duplicate rows.
            (Parallel across files.)

Memory: each MinHash signature is num_perm × 8 bytes. At num_perm=128 and
100M docs, signatures take ~100 GB — too large to hold all at once. This
script therefore builds the LSH index incrementally and only keeps a
signature until it is inserted; signatures of dropped docs are not stored.
Peak memory is roughly proportional to the number of *unique* docs kept.

For very large datasets (>50M docs), consider partitioned dedup or
using a distributed tool (e.g., text-dedup).

Dependency:
    pip install datasketch

Usage:
    python scripts/data_cleaning/dedup.py \\
        --input_dir  /scratch/.../quality_filtered \\
        --output_dir /scratch/.../deduped \\
        --threshold  0.8

    # More aggressive dedup
    python scripts/data_cleaning/dedup.py \\
        --input_dir  /scratch/.../quality_filtered \\
        --output_dir /scratch/.../deduped_strict \\
        --threshold  0.7 --num_perm 128
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

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    raise ImportError(
        "datasketch is required for dedup.py.\n"
        "Install with: pip install datasketch"
    )


# ---------------------------------------------------------------------------
# MinHash helper
# ---------------------------------------------------------------------------

def _minhash(text: str, num_perm: int, ngram: int = 5) -> MinHash:
    """Compute a MinHash signature over word n-grams."""
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    for i in range(max(1, len(words) - ngram + 1)):
        gram = " ".join(words[i : i + ngram])
        m.update(gram.encode("utf-8"))
    return m


# ---------------------------------------------------------------------------
# Pass 1: build LSH index and collect duplicate IDs (single process)
# ---------------------------------------------------------------------------

def _collect_duplicates(
    files: list[str],
    text_column: str,
    threshold: float,
    num_perm: int,
    ngram: int,
) -> set:
    """Return the set of IDs that are near-duplicates of an earlier doc."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    drop_ids: set = set()
    total = inserted = 0
    start = time.time()

    for file_path in files:
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(batch_size=256, columns=["id", text_column]):
            rows = pa.Table.from_batches([batch]).to_pylist()
            for row in rows:
                total += 1
                doc_id = str(row["id"])
                text   = row.get(text_column) or ""
                mh = _minhash(text, num_perm, ngram)
                results = lsh.query(mh)
                if results:
                    drop_ids.add(row["id"])
                else:
                    lsh.insert(doc_id, mh)
                    inserted += 1

        elapsed = time.time() - start
        print(
            f"  [pass1] {total:,} seen | {inserted:,} kept | "
            f"{len(drop_ids):,} dupes | {total / elapsed:,.0f} docs/s | "
            f"{os.path.basename(file_path)}"
        )

    elapsed = time.time() - start
    pct = 100 * len(drop_ids) / total if total else 0
    print(f"  [pass1] Done: {len(drop_ids):,} duplicates ({pct:.1f}%) in {elapsed:.1f}s")
    return drop_ids


# ---------------------------------------------------------------------------
# Pass 2: write filtered parquets (parallel)
# ---------------------------------------------------------------------------

def _write_filtered(args: tuple[str, str, set]) -> tuple[str, int, int]:
    file_path, output_dir, drop_ids = args
    out_path = os.path.join(output_dir, os.path.basename(file_path))
    pf = pq.ParquetFile(file_path)
    writer = None
    total = kept = 0

    for batch in pf.iter_batches(batch_size=512):
        tbl = pa.Table.from_batches([batch])
        total += len(tbl)
        mask = [sid not in drop_ids for sid in tbl.column("id").to_pylist()]
        filtered = tbl.filter(mask)
        kept += len(filtered)
        if len(filtered) > 0:
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
        description="Near-deduplication using MinHash LSH",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir",   required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--threshold",   type=float, default=0.8,
                        help="Jaccard similarity threshold for near-duplicates")
    parser.add_argument("--num_perm",    type=int, default=128,
                        help="MinHash permutations — higher = more accurate, slower")
    parser.add_argument("--ngram",       type=int, default=5,
                        help="Word n-gram size for shingling")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--glob",        default="train-*.parquet")
    parser.add_argument("--workers",     type=int, default=min(cpu_count(), 8),
                        help="Workers for pass 2 (pass 1 is always single-process)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.glob)))
    if not files:
        raise FileNotFoundError(f"No files matched '{args.glob}' in {args.input_dir}")
    print(f"Files     : {len(files)}")
    print(f"Threshold : {args.threshold}  num_perm={args.num_perm}  ngram={args.ngram}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Pass 1
    print("\nPass 1: building LSH index and collecting duplicate IDs ...")
    drop_ids = _collect_duplicates(
        files, args.text_column, args.threshold, args.num_perm, args.ngram
    )

    # Pass 2
    print(f"\nPass 2: writing {len(files)} filtered files ({args.workers} workers) ...")
    work = [(f, args.output_dir, drop_ids) for f in files]
    num_workers = min(args.workers, len(work))

    start = time.time()
    total_all = kept_all = 0

    with Pool(num_workers) as pool:
        for path, total, kept in pool.imap_unordered(_write_filtered, work):
            total_all += total
            kept_all  += kept
            elapsed = time.time() - start
            pct = 100 * kept_all / total_all if total_all else 0
            print(
                f"  {os.path.basename(path)} | {kept}/{total} | "
                f"running: {kept_all:,}/{total_all:,} ({pct:.1f}%) | {elapsed:.1f}s"
            )

    elapsed = time.time() - start
    pct = 100 * kept_all / total_all if total_all else 0
    print(f"\nDone: {kept_all:,} / {total_all:,} kept ({pct:.1f}%) in {elapsed:.1f}s")
    print(f"Output → {args.output_dir}")


if __name__ == "__main__":
    main()
