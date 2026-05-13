"""torchrun-aware entrypoint for the titan trainer.

Reads ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE`` from the env (set by
``torchrun``), initializes the NCCL process group, parses the TOML config,
and hands off to ``train.main``.

Single-process invocation (no torchrun) is also supported — useful for the
1-GPU pilot — and degenerates to a no-op process group setup.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make ``import umich_slm`` work when this is invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_PATH = _REPO_ROOT / "src"
if _SRC_PATH.exists() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

import torch
import torch.distributed as dist


def _init_distributed() -> tuple[int, int, int]:
    """Initialize NCCL if launched under torchrun, else single-process.

    Returns ``(rank, local_rank, world_size)``.
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UMichSLM torchtitan trainer launcher")
    parser.add_argument("config", type=str, help="Path to TOML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help=(
            "Override a TOML field; repeatable. Example: "
            "--set training.gradient_accumulation_steps=4 "
            "--set job.run_name=titan_baseline_ddp4"
        ),
    )
    args = parser.parse_args(argv)

    rank, local_rank, world_size = _init_distributed()

    # Imports happen after path setup so the package resolves cleanly.
    from umich_slm.titan.config_schema import apply_overrides, load_toml_config
    from torchtitan_train.train import run

    config = load_toml_config(args.config)
    if args.overrides:
        apply_overrides(config, args.overrides)
    try:
        # Multi-stage in-process driver vs. single-stage existing run.
        if getattr(config, "staging", None) and config.staging.mode == "in_process":
            from torchtitan_train.staging_loop import run_staged

            run_staged(config, rank=rank, local_rank=local_rank, world_size=world_size)
        else:
            run(config, rank=rank, local_rank=local_rank, world_size=world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
