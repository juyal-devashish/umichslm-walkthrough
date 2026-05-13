"""Checkpointing in HuggingFace safetensors format.

Saves rank-0 only because DDP keeps full parameter replicas. The output
directory layout matches what ``compute_val_loss.py``, ``compute_bpb.py``,
and ``run_eval.py`` expect: ``checkpoint-{step}/`` containing the standard
HF files plus a sidecar ``optimizer.pt`` (optimizer + scheduler + step + rng)
and ``trainer_state.json`` (mirrors HF Trainer's format so the existing
plotting / scaling tooling keeps working).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config_schema import TitanConfig
from .parallelize import is_rank_zero, unwrap_ddp


def _checkpoint_dir(output_dir: Path, step: int) -> Path:
    return output_dir / f"checkpoint-{step}"


def _serialize_config(config: TitanConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in (
        "job", "model", "data", "training", "optimizer",
        "lr_scheduler", "parallelism", "logging", "checkpoint",
    ):
        section = getattr(config, name, None)
        if section is not None:
            out[name] = asdict(section)
    return out


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    *,
    step: int,
    config: TitanConfig,
    log_history: Optional[list] = None,
) -> Optional[Path]:
    """Save an HF-compatible checkpoint at ``output_dir/checkpoint-{step}``.

    Returns the path on rank 0, ``None`` on other ranks. The trainer should
    barrier after this to keep all ranks in lockstep before continuing.
    """
    if not is_rank_zero():
        return None

    output_dir = Path(config.job.output_dir)
    ckpt_dir = _checkpoint_dir(output_dir, step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    inner = unwrap_ddp(model)
    # The wrapper's ``inner`` attribute is the HF ``UMichSLMForCausalLM``.
    hf_model = getattr(inner, "inner", inner)
    hf_model.save_pretrained(ckpt_dir, safe_serialization=True)

    # Optimizer/scheduler/step/RNG sidecar.
    sidecar: Dict[str, Any] = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if scheduler is not None:
        sidecar["scheduler"] = scheduler.state_dict()
    if torch.cuda.is_available():
        sidecar["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(sidecar, ckpt_dir / "optimizer.pt")

    # Resolved config snapshot (for reproducibility) + HF-style trainer_state.
    with open(ckpt_dir / "titan_config.json", "w") as fh:
        json.dump(_serialize_config(config), fh, indent=2, default=str)

    trainer_state = {
        "global_step": step,
        "max_steps": config.training.max_steps,
        "log_history": log_history or [],
    }
    with open(ckpt_dir / "trainer_state.json", "w") as fh:
        json.dump(trainer_state, fh, indent=2)

    _prune_old_checkpoints(output_dir, keep_last=config.checkpoint.keep_last)
    return ckpt_dir


def _prune_old_checkpoints(output_dir: Path, *, keep_last: int) -> None:
    if keep_last <= 0:
        return
    ckpts = sorted(
        (p for p in output_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    excess = len(ckpts) - keep_last
    for p in ckpts[: max(excess, 0)]:
        shutil.rmtree(p, ignore_errors=True)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    *,
    map_location: Optional[str | torch.device] = None,
) -> int:
    """Resume from a previously saved ``checkpoint-{step}`` directory.

    Loads the model weights via ``from_pretrained`` and the sidecar state.
    Returns the step number to continue from.
    """
    path = Path(path)
    inner = unwrap_ddp(model)
    hf_model = getattr(inner, "inner", inner)

    # Reload weights in-place to preserve any module hooks the wrapper added.
    state_dict_path = path / "model.safetensors"
    if not state_dict_path.exists():
        # Fall back to multi-shard or pytorch-bin layouts that HF supports.
        state_dict_path = None
    if state_dict_path is not None:
        from safetensors.torch import load_file

        sd = load_file(str(state_dict_path), device="cpu")
        missing, unexpected = hf_model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"Missing keys when resuming: {missing[:5]}...")
        if unexpected:
            raise RuntimeError(f"Unexpected keys when resuming: {unexpected[:5]}...")
    else:
        # Use HF loader for shard layouts.
        loaded = type(hf_model).from_pretrained(path)
        hf_model.load_state_dict(loaded.state_dict())

    sidecar_path = path / "optimizer.pt"
    if sidecar_path.exists() and (optimizer is not None or scheduler is not None):
        sidecar = torch.load(sidecar_path, map_location=map_location or "cpu", weights_only=False)
        if optimizer is not None and "optimizer" in sidecar:
            optimizer.load_state_dict(sidecar["optimizer"])
        if scheduler is not None and "scheduler" in sidecar:
            scheduler.load_state_dict(sidecar["scheduler"])
        if "torch_rng_state" in sidecar:
            torch.set_rng_state(sidecar["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in sidecar:
            torch.cuda.set_rng_state_all(sidecar["cuda_rng_state_all"])
        return int(sidecar.get("step", 0))

    return 0
