#!/usr/bin/env python3
"""Convenience wrapper for the torchtitan-style trainer.

Mirrors ``scripts/llamafactory_train.py``: ensures the project root is on
``sys.path`` so ``import umich_slm`` resolves, registers the model with HF
Auto classes, and dispatches to ``torchtitan_train.launch:main``.

Single-process and torchrun usage:

    # Single GPU
    python scripts/torchtitan_train.py configs/torchtitan/umichslm_pretrain_baseline.toml

    # Multi-GPU (single node)
    torchrun --standalone --nproc_per_node=4 \\
        scripts/torchtitan_train.py configs/torchtitan/umichslm_pretrain_baseline.toml
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Trigger AutoConfig/AutoModel registration in this process and any spawned
# dataloader workers (PYTHONPATH propagates to them under torchrun).
import umich_slm  # noqa: F401

from torchtitan_train.launch import main as launch_main


if __name__ == "__main__":
    # Make sure subprocesses (dataloader workers, distributed children) can
    # also import the package.
    pythonpath = os.environ.get("PYTHONPATH", "")
    if str(_REPO_ROOT) not in pythonpath:
        os.environ["PYTHONPATH"] = (
            f"{_REPO_ROOT}{os.pathsep}{pythonpath}" if pythonpath else str(_REPO_ROOT)
        )
    raise SystemExit(launch_main(sys.argv[1:]))
