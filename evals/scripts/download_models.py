"""Download baseline models to the shared HF cache before running evals.

Run this once on a GPU node (or login node with internet access) before
submitting the SLURM eval job:

    python evals/scripts/download_models.py
"""

import os
from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen3-0.6B",
]

HF_HOME = os.environ.get(
    "HF_HOME",
    "/path/to/data",
)
os.environ["HF_HOME"] = HF_HOME

print(f"Using HF_HOME: {HF_HOME}")

for model_id in MODELS:
    print(f"\nDownloading {model_id} ...")
    path = snapshot_download(repo_id=model_id)
    print(f"  Cached at: {path}")

print("\nAll models downloaded.")
