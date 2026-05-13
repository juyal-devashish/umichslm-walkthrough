"""Single-node DDP pretraining loop for UMichSLM, torchtitan-style.

Mirrors what ``torchtitan/train.py`` does in spirit (TOML config, MFU
reporting, periodic checkpoints) but stays self-contained — no torchtitan
parallelize calls, no FSDP2/TP/PP. The model adapter
(``umich_slm.titan.UMichSLMTitanWrapper``) carries the loss fusion;
this module wires the rest: optimizer, LR schedule, DDP, dataloader, wandb,
metrics.jsonl, and HF-format checkpoints.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Path setup duplicated here so this module also works as a direct entrypoint.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_PATH = _REPO_ROOT / "src"
if _SRC_PATH.exists() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

import umich_slm  # noqa: F401  - registers AutoConfig/AutoModel for HF
from umich_slm.titan import (
    TitanConfig,
    UMichSLMTitanWrapper,
    build_wrapper_from_config,
)
from umich_slm.titan.checkpoint import load_checkpoint, save_checkpoint
from umich_slm.titan.data import build_dataloader
from umich_slm.titan.losses import detach_scalar, loss_components
from umich_slm.titan.parallelize import (
    apply_ddp,
    barrier,
    get_rank,
    is_rank_zero,
    unwrap_ddp,
)
from umich_slm.titan.throughput import (
    HW_PEAK_TFLOPS,
    FlopsCounter,
    MFUTracker,
    StepTimer,
    detect_hardware_key,
    gpu_inventory_snapshot,
)


# ----------------------------------------------------------------- LR schedule
def _make_lr_lambda(cfg) -> Any:
    """Return a callable suitable for ``LambdaLR``.

    Supports two modes: ``cosine`` (warmup + cosine) and
    ``warmup_stable_decay`` (warmup + flat + cosine/linear decay), matching
    LLaMA Factory's options.
    """
    name = cfg.name.lower()
    warmup = max(int(cfg.warmup_steps), 0)
    stable = max(int(cfg.stable_steps), 0)
    decay = max(int(cfg.decay_steps), 0)
    min_ratio = float(cfg.min_lr_ratio)
    decay_type = cfg.decay_type.lower()

    if name == "cosine":
        # Treat ``warmup + stable + decay`` as the total horizon for cosine.
        total = warmup + stable + decay

        def lr_cosine(step: int) -> float:
            if total <= 0:
                return 1.0
            if step < warmup:
                return (step + 1) / max(warmup, 1)
            progress = (step - warmup) / max(total - warmup, 1)
            cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return min_ratio + (1.0 - min_ratio) * cos

        return lr_cosine

    if name == "warmup_stable_decay":
        decay_start = warmup + stable

        def lr_wsd(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(warmup, 1)
            if step < decay_start:
                return 1.0
            progress = (step - decay_start) / max(decay, 1)
            progress = min(progress, 1.0)
            if decay_type == "linear":
                return min_ratio + (1.0 - min_ratio) * (1.0 - progress)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cos

        return lr_wsd

    raise ValueError(f"Unknown lr_scheduler.name: {cfg.name}")


# ------------------------------------------------------------------ optimizer
def _make_optimizer(wrapper: UMichSLMTitanWrapper, cfg) -> torch.optim.Optimizer:
    groups = wrapper.param_groups(weight_decay=cfg.weight_decay)
    name = cfg.name.lower()
    fused = name == "adamw_fused" and torch.cuda.is_available()
    return AdamW(
        groups,
        lr=cfg.lr,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
        fused=fused,
    )


# --------------------------------------------------------------------- logger
class JsonlLogger:
    """Append-only JSON lines on rank 0; no-op on other ranks."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate on init so resumed runs append cleanly to a fresh file
            # in the new output dir; existing histories live alongside the
            # checkpoint they were written with.
            path.write_text("")

    def log(self, payload: Dict[str, Any]) -> None:
        if self.path is None:
            return
        with open(self.path, "a") as fh:
            fh.write(json.dumps(payload) + "\n")


def _maybe_init_wandb(config: TitanConfig, rank: int, hw_info: Dict[str, str]):
    if rank != 0 or config.logging.report_to.lower() != "wandb":
        return None
    try:
        import wandb
    except ImportError:
        print("[titan] wandb not installed; falling back to metrics.jsonl only", flush=True)
        return None
    serialized: Dict[str, Any] = {}
    for name in ("job", "model", "data", "training", "optimizer", "lr_scheduler",
                 "parallelism", "logging", "checkpoint"):
        section = getattr(config, name, None)
        if section is not None:
            serialized[name] = asdict(section)
    serialized["_hw"] = hw_info
    run = wandb.init(
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        name=config.job.run_name,
        config=serialized,
        tags=config.logging.wandb_tags or None,
        dir=config.job.output_dir,
        reinit=True,
    )
    return run


# ------------------------------------------------------------------- helpers
def _resolve_arrow_path(dataset_dir: str, leaf: str) -> Path:
    p = Path(leaf)
    if p.is_absolute():
        return p
    return Path(dataset_dir) / leaf


def _load_tokenizer(config: TitanConfig):
    path = config.data.tokenizer_path or config.model.init_from
    if not path:
        raise ValueError(
            "Tokenizer path required: set [data].tokenizer_path or "
            "[model].init_from to a checkpoint that bundles a tokenizer."
        )
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _next_batch(iterator, dataloader_factory):
    """Pull the next batch, restarting the iterator if the dataloader is finite."""
    try:
        return next(iterator), iterator
    except StopIteration:
        new_iter = iter(dataloader_factory())
        return next(new_iter), new_iter


def _print_rank0(msg: str) -> None:
    if is_rank_zero():
        print(msg, flush=True)


# ------------------------------------------------------------------ training
def run(config: TitanConfig, *, rank: int, local_rank: int, world_size: int) -> None:
    output_dir = Path(config.job.output_dir)
    if is_rank_zero():
        output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.job.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.job.seed + rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Tokenizer first — it determines the model's vocab_size.
    tokenizer = _load_tokenizer(config)
    vocab_size = len(tokenizer)
    if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size:
        # Use the larger of len() (covers added tokens) and vocab_size.
        vocab_size = max(vocab_size, int(tokenizer.vocab_size))
    _print_rank0(f"[titan] tokenizer={tokenizer.__class__.__name__} vocab_size={vocab_size}")

    # Model.
    wrapper = build_wrapper_from_config(config.model, vocab_size=vocab_size)
    wrapper = wrapper.to(device)

    if config.training.gradient_checkpointing:
        wrapper.apply_gradient_checkpointing()
    # Order matters: float8 must run before compile so the captured graph
    # reflects the fp8 GEMMs. Float8 also requires Hopper+, so apply_float8
    # raises a clear error on older GPUs.
    if config.training.float8:
        converted = wrapper.apply_float8()
        _print_rank0(f"[titan] float8 conversion applied to {converted} linear layers")
    if config.training.torch_compile:
        compiled = wrapper.apply_compile(
            deltanet_compile_safe=config.training.deltanet_compile_safe
        )
        _print_rank0(f"[titan] torch.compile applied to {compiled} decoder blocks")

    # Numerical precision: mixed bf16 via autocast (no GradScaler needed).
    autocast_dtype = torch.bfloat16 if config.training.bf16 else torch.float32
    use_autocast = config.training.bf16 and device.type == "cuda"

    n_params = wrapper.num_params
    n_active = wrapper.num_active_params
    _print_rank0(
        f"[titan] params total={n_params/1e6:.2f}M active={n_active/1e6:.2f}M "
        f"size={config.model.size}"
    )

    model = apply_ddp(wrapper, local_rank=local_rank)

    # Optimizer + scheduler.
    optimizer = _make_optimizer(wrapper, config.optimizer)
    lr_lambda = _make_lr_lambda(config.lr_scheduler)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Data.
    train_arrow = _resolve_arrow_path(config.data.dataset_dir, config.data.train_arrow)
    val_arrow = _resolve_arrow_path(config.data.dataset_dir, config.data.val_arrow)

    def make_train_loader():
        return build_dataloader(
            train_arrow,
            tokenizer,
            seq_len=config.data.seq_len,
            batch_size=config.training.per_device_batch_size,
            text_column=config.data.text_column,
            rank=rank,
            world_size=world_size,
            seed=config.job.seed,
            num_workers=config.data.num_workers,
            infinite=True,
        )

    train_loader = make_train_loader()
    train_iter = iter(train_loader)

    # Resume.
    start_step = 0
    if config.checkpoint.resume_from:
        start_step = load_checkpoint(
            Path(config.checkpoint.resume_from),
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        _print_rank0(f"[titan] resumed from {config.checkpoint.resume_from} at step {start_step}")

    # Throughput infra.
    hw_key = detect_hardware_key("bf16" if config.training.bf16 else "fp32")
    peak_per_gpu = HW_PEAK_TFLOPS.get(hw_key, 0.0)
    if peak_per_gpu == 0.0:
        _print_rank0(f"[titan] WARNING: no peak FLOPs entry for hw_key={hw_key}; MFU will be 0")
    aggregate_peak_tflops = peak_per_gpu * max(world_size, 1)
    flops_counter = FlopsCounter(unwrap_ddp(model).config, seq_len=config.data.seq_len)
    flops_counter.set_active_params(n_active)
    mfu = MFUTracker(peak_tflops=aggregate_peak_tflops)

    timer = StepTimer(device=device)
    hw_info = {**gpu_inventory_snapshot(), "hw_key": hw_key, "world_size": str(world_size)}

    # Loggers.
    metrics_path = output_dir / config.logging.metrics_jsonl if is_rank_zero() else None
    jsonl = JsonlLogger(metrics_path)
    wandb_run = _maybe_init_wandb(config, rank, hw_info)

    # Training loop.
    grad_accum = max(int(config.training.gradient_accumulation_steps), 1)
    tokens_per_step = (
        world_size
        * config.training.per_device_batch_size
        * grad_accum
        * config.data.seq_len
    )
    log_history: List[Dict[str, Any]] = []
    model.train()

    _print_rank0(
        f"[titan] starting training "
        f"max_steps={config.training.max_steps} "
        f"tokens_per_step={tokens_per_step:,} "
        f"world_size={world_size} bf16={config.training.bf16}"
    )

    for step in range(start_step, config.training.max_steps):
        timer.start()
        optimizer.zero_grad(set_to_none=True)

        accum_loss = 0.0
        component_accum: Dict[str, float] = {"lm_loss": 0.0, "mtp_loss": 0.0, "aux_loss": 0.0}
        component_counts: Dict[str, int] = {"lm_loss": 0, "mtp_loss": 0, "aux_loss": 0}

        for micro in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(make_train_loader())
                batch = next(train_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            no_sync = (
                model.no_sync() if hasattr(model, "no_sync") and micro < grad_accum - 1
                else nullcontext()
            )
            ctx = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if use_autocast
                else nullcontext()
            )
            with no_sync:
                with ctx:
                    output = model(input_ids, labels=labels)
                loss = output.loss / grad_accum
                loss.backward()
                accum_loss += float(loss.detach().float().item())

                comps = loss_components(output)
                for k in component_accum:
                    v = comps.get(k)
                    if v is not None:
                        component_accum[k] += v
                        component_counts[k] += 1

        grad_norm = torch.nn.utils.clip_grad_norm_(
            unwrap_ddp(model).parameters(), config.training.max_grad_norm
        )
        optimizer.step()
        scheduler.step()

        elapsed_ms, peak_bytes = timer.end()
        elapsed_s = elapsed_ms / 1000.0
        tokens_per_sec = tokens_per_step / elapsed_s if elapsed_s > 0 else 0.0
        flops_total = flops_counter.per_step_flops(tokens_per_step=tokens_per_step)
        mfu_pct = mfu.mfu(total_flops=flops_total, elapsed_seconds=elapsed_s) * 100.0

        if (step + 1) % max(config.logging.log_every, 1) == 0 or step == start_step:
            avg_components = {
                k: (component_accum[k] / component_counts[k]) if component_counts[k] else None
                for k in component_accum
            }
            payload = {
                "step": step + 1,
                "loss": accum_loss * grad_accum,  # un-scale to per-micro mean
                "grad_norm": float(grad_norm.detach().float().item())
                if hasattr(grad_norm, "detach") else float(grad_norm),
                "lr": float(scheduler.get_last_lr()[0]),
                "step_time_ms": elapsed_ms,
                "tokens_per_sec": tokens_per_sec,
                "mfu_pct": mfu_pct,
                "peak_mem_gb": peak_bytes / (1024 ** 3),
                **{f"component/{k}": v for k, v in avg_components.items()},
            }
            log_history.append(payload)
            if is_rank_zero():
                jsonl.log(payload)
                if wandb_run is not None:
                    wandb_run.log(payload, step=step + 1)
                _print_rank0(
                    f"[titan] step={step+1}/{config.training.max_steps} "
                    f"loss={payload['loss']:.4f} lr={payload['lr']:.2e} "
                    f"tok/s={tokens_per_sec:,.0f} mfu={mfu_pct:.1f}% "
                    f"step_ms={elapsed_ms:.1f} mem={payload['peak_mem_gb']:.1f}GB"
                )

        # Periodic checkpoint.
        save_every = max(int(config.checkpoint.save_every), 0)
        if save_every and (step + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler,
                step=step + 1, config=config, log_history=log_history,
            )
            barrier()

    # Final checkpoint.
    save_checkpoint(
        model, optimizer, scheduler,
        step=config.training.max_steps, config=config, log_history=log_history,
    )
    barrier()
    if wandb_run is not None:
        wandb_run.finish()
    _print_rank0("[titan] training complete")
