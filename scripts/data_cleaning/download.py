"""Download pretraining datasets from HuggingFace Hub.

Consolidates all dataset downloads behind a single CLI. Uses snapshot_download
for efficient parallel fetching and resume support.

Supported datasets:
  smollm_corpus      — HuggingFaceTB/smollm-corpus (fineweb-edu-dedup,
                        cosmopedia-v2, python-edu subsets)
  fineweb_edu        — HuggingFaceFW/fineweb-edu
  dclm               — mlfoundations/dclm-baseline-1.0
  opencode_reasoning — nvidia/OpenCodeReasoning
  fb_reasoning       — facebook/natural_reasoning

Usage:
    # Download the main pretraining corpus (FineWeb-Edu-Dedup subset only)
    python scripts/data_cleaning/download.py \\
        --dataset smollm_corpus \\
        --subset fineweb-edu-dedup \\
        --local_dir /path/to/data

    # Download everything in smollm-corpus
    python scripts/data_cleaning/download.py \\
        --dataset smollm_corpus \\
        --local_dir /path/to/data

    # Download DCLM baseline
    python scripts/data_cleaning/download.py \\
        --dataset dclm \\
        --local_dir /path/to/data \\
        --max_workers 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "smollm_corpus": {
        "repo_id": "HuggingFaceTB/smollm-corpus",
        "subsets": ["fineweb-edu-dedup", "cosmopedia-v2", "python-edu"],
        "default_subset": None,  # None means download all
    },
    "fineweb_edu": {
        "repo_id": "HuggingFaceFW/fineweb-edu",
        "subsets": [],
        "default_subset": None,
    },
    "dclm": {
        "repo_id": "mlfoundations/dclm-baseline-1.0",
        "subsets": [],
        "default_subset": None,
    },
    "opencode_reasoning": {
        "repo_id": "nvidia/OpenCodeReasoning",
        "subsets": [],
        "default_subset": None,
    },
    "fb_reasoning": {
        "repo_id": "facebook/natural_reasoning",
        "subsets": [],
        "default_subset": None,
    },
}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def download(
    dataset: str,
    local_dir: str,
    cache_dir: str | None = None,
    subset: str | None = None,
    max_workers: int = 8,
    dry_run: bool = False,
) -> None:
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choices: {', '.join(DATASETS)}"
        )
    spec = DATASETS[dataset]
    repo_id = spec["repo_id"]

    # Validate subset if provided
    if subset and spec["subsets"] and subset not in spec["subsets"]:
        raise ValueError(
            f"Unknown subset '{subset}' for {dataset}. "
            f"Choices: {', '.join(spec['subsets'])}"
        )

    # Build allow_patterns to download only the requested subset directory
    allow_patterns: list[str] | None = None
    if subset:
        allow_patterns = [f"data/{subset}/*", f"{subset}/*", "*.json", "*.md"]

    print(f"Dataset  : {repo_id}")
    print(f"Subset   : {subset or '(all)'}")
    print(f"Local dir: {local_dir}")
    print(f"Cache dir: {cache_dir or '(default HF cache)'}")
    print(f"Workers  : {max_workers}")
    if allow_patterns:
        print(f"Patterns : {allow_patterns}")

    if dry_run:
        print("[DRY RUN] Skipping download.")
        return

    Path(local_dir).mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        max_workers=max_workers,
    )
    print(f"Done. Files saved to {local_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pretraining datasets from HuggingFace Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASETS),
        help="Dataset to download",
    )
    parser.add_argument(
        "--local_dir", required=True,
        help="Local directory to save the dataset",
    )
    parser.add_argument(
        "--cache_dir", default=None,
        help="HuggingFace cache directory (defaults to ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--subset", default=None,
        help="Subset/config to download (smollm_corpus: fineweb-edu-dedup, cosmopedia-v2, python-edu)",
    )
    parser.add_argument(
        "--max_workers", type=int, default=8,
        help="Parallel download workers",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be downloaded without actually downloading",
    )
    args = parser.parse_args()
    download(**vars(args))


if __name__ == "__main__":
    main()
