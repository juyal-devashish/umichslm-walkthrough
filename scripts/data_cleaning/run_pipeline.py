"""End-to-end data cleaning pipeline for FineWeb-Edu (or similar parquet corpora).

Chains four steps in order:
  1. extract_scores   — build compact (id, score, file_path) score map
  2. filter_by_score  — keep docs with score >= threshold
  3. text_quality     — heuristic per-doc filters (length, repetition, chars)
  4. dedup            — MinHash near-deduplication

Each step is invoked as a subprocess so it can also be run standalone.
All parameters can be set via a JSON config file; CLI flags override the config.

Config file format (all keys optional — defaults shown):
    {
      "data_dir":           "/path/to/data",
      "work_dir":           "/path/to/data",
      "score_threshold":    3.0,
      "min_words":          50,
      "max_words":          100000,
      "max_nonalpha_ratio": 0.40,
      "max_digit_ratio":    0.20,
      "max_line_rep_ratio": 0.30,
      "max_bullet_rep_ratio": 0.50,
      "max_short_line_ratio": 0.60,
      "dedup_threshold":    0.8,
      "dedup_num_perm":     128,
      "workers":            8
    }

Usage:
    # Full pipeline with a config file
    python scripts/data_cleaning/run_pipeline.py \\
        --config configs/data_cleaning.json

    # Run only specific steps (skip download if data already on disk)
    python scripts/data_cleaning/run_pipeline.py \\
        --config configs/data_cleaning.json \\
        --steps extract_scores filter_by_score

    # Quick inline override
    python scripts/data_cleaning/run_pipeline.py \\
        --config configs/data_cleaning.json \\
        --score_threshold 4.0 --workers 4

    # Dry run: print commands without executing
    python scripts/data_cleaning/run_pipeline.py \\
        --config configs/data_cleaning.json \\
        --dry_run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS: dict = {
    "score_threshold":      3.0,
    "min_words":            50,
    "max_words":            100_000,
    "max_nonalpha_ratio":   0.40,
    "max_digit_ratio":      0.20,
    "max_line_rep_ratio":   0.30,
    "max_bullet_rep_ratio": 0.50,
    "max_short_line_ratio": 0.60,
    "dedup_threshold":      0.8,
    "dedup_num_perm":       128,
    "workers":              8,
    "glob":                 "train-*.parquet",
}

ALL_STEPS = ["extract_scores", "filter_by_score", "text_quality", "dedup"]

# ---------------------------------------------------------------------------
# Step builders — return the subprocess command list
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def _cmd_extract_scores(cfg: dict) -> list[str]:
    score_map = str(Path(cfg["work_dir"]) / "score_map.parquet")
    return [
        sys.executable, str(SCRIPT_DIR / "extract_scores.py"),
        "--data_dir", cfg["data_dir"],
        "--output",   score_map,
        "--workers",  str(cfg["workers"]),
        "--glob",     cfg["glob"],
    ]


def _cmd_filter_by_score(cfg: dict) -> list[str]:
    score_map    = str(Path(cfg["work_dir"]) / "score_map.parquet")
    output_dir   = str(Path(cfg["work_dir"]) / "01_score_filtered")
    return [
        sys.executable, str(SCRIPT_DIR / "filter_by_score.py"),
        "--score_map",  score_map,
        "--data_dir",   cfg["data_dir"],
        "--output_dir", output_dir,
        "--threshold",  str(cfg["score_threshold"]),
        "--workers",    str(cfg["workers"]),
        "--glob",       cfg["glob"],
    ]


def _cmd_text_quality(cfg: dict) -> list[str]:
    input_dir  = str(Path(cfg["work_dir"]) / "01_score_filtered")
    output_dir = str(Path(cfg["work_dir"]) / "02_quality_filtered")
    return [
        sys.executable, str(SCRIPT_DIR / "text_quality_filters.py"),
        "--input_dir",           input_dir,
        "--output_dir",          output_dir,
        "--workers",             str(cfg["workers"]),
        "--glob",                cfg["glob"],
        "--min_words",           str(cfg["min_words"]),
        "--max_words",           str(cfg["max_words"]),
        "--max_nonalpha_ratio",  str(cfg["max_nonalpha_ratio"]),
        "--max_digit_ratio",     str(cfg["max_digit_ratio"]),
        "--max_line_rep_ratio",  str(cfg["max_line_rep_ratio"]),
        "--max_bullet_rep_ratio",str(cfg["max_bullet_rep_ratio"]),
        "--max_short_line_ratio",str(cfg["max_short_line_ratio"]),
    ]


def _cmd_dedup(cfg: dict) -> list[str]:
    input_dir  = str(Path(cfg["work_dir"]) / "02_quality_filtered")
    output_dir = str(Path(cfg["work_dir"]) / "03_deduped")
    return [
        sys.executable, str(SCRIPT_DIR / "dedup.py"),
        "--input_dir",  input_dir,
        "--output_dir", output_dir,
        "--threshold",  str(cfg["dedup_threshold"]),
        "--num_perm",   str(cfg["dedup_num_perm"]),
        "--workers",    str(cfg["workers"]),
        "--glob",       cfg["glob"],
    ]


STEP_COMMANDS = {
    "extract_scores":  _cmd_extract_scores,
    "filter_by_score": _cmd_filter_by_score,
    "text_quality":    _cmd_text_quality,
    "dedup":           _cmd_dedup,
}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_step(name: str, cmd: list[str], dry_run: bool) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"STEP: {name}")
    print(f"{sep}")
    print("Command:", " ".join(cmd))
    if dry_run:
        print("[DRY RUN] skipping execution")
        return
    t0 = time.time()
    result = subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[ERROR] Step '{name}' failed (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"[OK] {name} finished in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end data cleaning pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None,
                        help="Path to JSON config file")
    parser.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS,
                        help="Steps to run (default: all)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    # Required if not in config
    parser.add_argument("--data_dir",  default=None, help="Raw parquet directory")
    parser.add_argument("--work_dir",  default=None, help="Working directory for pipeline outputs")
    # Overridable parameters
    parser.add_argument("--score_threshold",      type=float, default=None)
    parser.add_argument("--min_words",            type=int,   default=None)
    parser.add_argument("--max_words",            type=int,   default=None)
    parser.add_argument("--max_nonalpha_ratio",   type=float, default=None)
    parser.add_argument("--max_digit_ratio",      type=float, default=None)
    parser.add_argument("--max_line_rep_ratio",   type=float, default=None)
    parser.add_argument("--max_bullet_rep_ratio", type=float, default=None)
    parser.add_argument("--max_short_line_ratio", type=float, default=None)
    parser.add_argument("--dedup_threshold",      type=float, default=None)
    parser.add_argument("--dedup_num_perm",       type=int,   default=None)
    parser.add_argument("--workers",              type=int,   default=None)
    parser.add_argument("--glob",                 default=None)
    args = parser.parse_args()

    # Build config: defaults → JSON file → CLI overrides
    cfg: dict = dict(DEFAULTS)
    if args.config:
        file_cfg = json.loads(Path(args.config).read_text())
        cfg.update(file_cfg)

    cli_overrides = {
        k: v for k, v in vars(args).items()
        if v is not None and k not in ("config", "steps", "dry_run")
    }
    cfg.update(cli_overrides)

    if not cfg.get("data_dir"):
        parser.error("--data_dir is required (either via --config or CLI)")
    if not cfg.get("work_dir"):
        parser.error("--work_dir is required (either via --config or CLI)")

    Path(cfg["work_dir"]).mkdir(parents=True, exist_ok=True)

    # Save resolved config for reproducibility
    resolved_path = Path(cfg["work_dir"]) / "pipeline_config.json"
    if not args.dry_run:
        resolved_path.write_text(json.dumps(cfg, indent=2))
        print(f"Resolved config saved to {resolved_path}")

    print(f"\nPipeline steps: {args.steps}")
    print(f"Data dir : {cfg['data_dir']}")
    print(f"Work dir : {cfg['work_dir']}")

    pipeline_start = time.time()
    for step in args.steps:
        cmd = STEP_COMMANDS[step](cfg)
        _run_step(step, cmd, args.dry_run)

    elapsed = time.time() - pipeline_start
    final_dir = Path(cfg["work_dir"]) / "03_deduped"
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    if "dedup" in args.steps and not args.dry_run:
        print(f"Final cleaned data → {final_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
