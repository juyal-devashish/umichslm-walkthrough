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

import time

import torch
import torch.distributed as dist


def _init_distributed() -> tuple[int, int, int]:
    """Initialize NCCL if launched under torchrun, else single-process.

    Returns ``(rank, local_rank, world_size)``.
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # TITAN_DDP_SINGLE_RANK=1 wraps a 1-rank run in DDP anyway (NCCL, world 1): the
    # all-reduce is then a no-op, so a same-card A/B against a plain 1-rank run
    # isolates DDP's own bookkeeping (hooks, bucket copies, the unused-parameter
    # walk) from communication. Bench use only.
    force = os.environ.get("TITAN_DDP_SINGLE_RANK") == "1" and "MASTER_ADDR" in os.environ
    if world_size > 1 or force:
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        _set_device_with_retry(local_rank)
    return rank, local_rank, world_size


# How long to keep retrying CUDA device acquisition, and how long to wait between
# tries. Sized against the observed failure: a job at NICE=200 is backfilled the
# instant a card frees, which is exactly when the previous tenant may still be
# tearing its context down -- so `cudaSetDevice` returns cudaErrorDevicesUnavailable
# ("CUDA-capable device(s) is/are busy or unavailable") on a card that is healthy
# and idle seconds later. Two ar10m cells (55736, 55960) died this way inside three
# minutes on a node whose free GPU showed 1 MiB used, Default compute mode and a
# clean ECC/remap record. A genuinely dead card still fails, just 2 minutes later.
_CUDA_ACQUIRE_TRIES = 12
_CUDA_ACQUIRE_SLEEP_S = 10


def _set_device_with_retry(local_rank: int) -> None:
    """``torch.cuda.set_device`` with bounded retry on transient unavailability.

    Only retries the "busy or unavailable" class of failure; anything else (a bad
    ordinal, a driver mismatch) is a real error and is raised immediately rather
    than hidden behind two minutes of sleeping.
    """
    last: BaseException | None = None
    for attempt in range(1, _CUDA_ACQUIRE_TRIES + 1):
        try:
            torch.cuda.set_device(local_rank)
            if attempt > 1:
                print(
                    f"[launch] acquired CUDA device {local_rank} on attempt {attempt}"
                    f" after {(attempt - 1) * _CUDA_ACQUIRE_SLEEP_S}s",
                    flush=True,
                )
            return
        except Exception as exc:  # noqa: BLE001 - re-raised below if not transient
            if "busy or unavailable" not in str(exc):
                raise
            last = exc
            if attempt < _CUDA_ACQUIRE_TRIES:
                print(
                    f"[launch] CUDA device {local_rank} busy (attempt {attempt}/"
                    f"{_CUDA_ACQUIRE_TRIES}); retrying in {_CUDA_ACQUIRE_SLEEP_S}s",
                    flush=True,
                )
                time.sleep(_CUDA_ACQUIRE_SLEEP_S)
    raise RuntimeError(
        f"could not acquire CUDA device {local_rank} after "
        f"{_CUDA_ACQUIRE_TRIES * _CUDA_ACQUIRE_SLEEP_S}s of retries"
    ) from last


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
