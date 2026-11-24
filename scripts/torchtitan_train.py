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
_SRC = _REPO_ROOT / "src"


def _prepend(path: Path) -> None:
    """Put ``path`` at the FRONT of ``sys.path``, moving it if already present.

    The ordering is load-bearing, not cosmetic. For ``python
    scripts/torchtitan_train.py`` the interpreter makes ``scripts/`` into
    ``sys.path[0]``, and this file's own name collides with the top-level
    ``torchtitan_train`` PACKAGE -- the only such collision in the repo. If
    ``scripts/`` stays in front, ``from torchtitan_train.launch import main``
    re-imports *this file* as a module and dies with
    ``ModuleNotFoundError: ... 'torchtitan_train' is not a package``.

    A ``not in sys.path`` guard is not enough, and that is exactly how this
    shipped broken: in the ``umich-slm-fa2`` env on the GPU nodes the repo root
    is ALREADY on ``sys.path`` (pip-installed package plus an inherited
    PYTHONPATH), so the guard found it, skipped the insert, and left
    ``scripts/`` in front. Jobs 61674 and 61956 both lost their whole
    allocation to it after a clean preflight, while a plain ``python
    scripts/torchtitan_train.py`` on a host without that env works -- which is
    what made it look like a cluster problem rather than a repo bug.
    """
    while str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))


_prepend(_REPO_ROOT)
if _SRC.exists():
    _prepend(_SRC)

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
