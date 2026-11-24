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
from umich_slm.callbacks import MatFFNWarmupController, TSTController
from umich_slm.modules.mixture_of_depths import (
    gather_width_payload,
    gather_width_warning,
)
from umich_slm.modules.recurrent_depth import sample_recursion_count, seed_recursion_rng
from umich_slm.modules.swiglu import mat_ffn_occupancy_payload, step_route_thresholds
from umich_slm.titan import (
    TitanConfig,
    UMichSLMTitanWrapper,
    build_wrapper_from_config,
)
from umich_slm.titan.checkpoint import load_checkpoint, load_init_weights, save_checkpoint
from umich_slm.titan.compile_health import log_compile_health
from umich_slm.titan.data import build_dataloader
from umich_slm.titan.grad_guard import GradSpikeGuard
from umich_slm.titan.harmony_annotate import AnnotateOptions
from umich_slm.titan.losses import (
    TokenStatAccumulator,
    detach_scalar,
    loss_components,
    token_targets_from_batch,
)
from umich_slm.titan.sft_data import (
    IGNORE_INDEX,
    accum_weights,
    build_sft_dataloader,
)
from umich_slm.titan.parallelize import (
    apply_ddp,
    apply_fsdp,
    barrier,
    clip_grad_norm_,
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
from umich_slm.utils.mat_ffn_decay import apply_frequency_scaled_decay


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


# Parameters a warm start RE-COMPOSES rather than borrows: the tokenizer swap
# rebuilds both tables, while everything else comes from the donor body. Shared
# by embedding-healing (freeze_body_steps) and the borrowed-LR ramp.
_EMB_NAMES = ("embed_tokens", "lm_head")


def _borrowed_lr_mult(step: int, mult: float, hold: int, ramp: int) -> float:
    """Multiplier on the borrowed body's scheduled LR: ``mult`` for ``hold`` steps,
    then log-linear to 1.0 over ``ramp`` steps (equal ratio per step, so the
    climb from 0.01 spends as long below 0.1 as above it), then 1.0."""
    if step < hold:
        return mult
    if ramp <= 0 or step >= hold + ramp:
        return 1.0
    return mult ** (1.0 - (step - hold) / ramp)


def _split_borrowed_groups(
    groups: List[Dict[str, Any]], wrapper: UMichSLMTitanWrapper
) -> List[Dict[str, Any]]:
    """Split each param group into its re-composed part and its borrowed part,
    the latter tagged ``borrowed=True`` for the scheduler. Re-composed first, so
    group 0 (what the ``lr`` log key reads) stays the plain schedule."""
    emb_ids = {id(p) for n, p in wrapper.named_parameters() if any(k in n for k in _EMB_NAMES)}
    split: List[Dict[str, Any]] = []
    for g in groups:
        new = [p for p in g["params"] if id(p) in emb_ids]
        old = [p for p in g["params"] if id(p) not in emb_ids]
        if new:
            split.append({**g, "params": new})
        if old:
            split.append({**g, "params": old, "borrowed": True})
    return split


# ------------------------------------------------------------------ optimizer
def _make_optimizer(
    wrapper: UMichSLMTitanWrapper, cfg, split_borrowed: bool = False
) -> torch.optim.Optimizer:
    name = cfg.name.lower()
    if split_borrowed and name not in ("adamw", "adamw_fused"):
        raise ValueError(
            f"training.borrowed_lr_mult is an AdamW-path instrument; optimizer.name={name!r}"
        )
    # Delphi recipe: AdamH with Hyperball projection. Activated by
    # [optimizer].name = "adamh" in the TOML. The (tokens, batch) terms in
    # the recipe's β₂/ε/LR are read from [recipe] in train.py (caller); here
    # we honor whatever lr/betas/eps the caller computed.
    if name == "adamh":
        from umich_slm.optim.adamh import AdamH, split_param_groups

        # split_param_groups returns [proj (Hyperball), other (embed/norm/bias)]
        # and deliberately carries NO learning rate — the Delphi recipe uses a
        # distinct η per group. Passing a single ``lr`` to both (as this branch
        # used to) trains embeddings/norms/biases at η_proj, ~7.5x the intended
        # η_adam. LambdaLR scales each group's own base_lr, so the WSD schedule
        # rides on top of both correctly.
        groups = split_param_groups(wrapper.inner)
        groups[0]["lr"] = cfg.lr
        groups[1]["lr"] = cfg.lr_adam if cfg.lr_adam is not None else cfg.lr
        return AdamH(groups, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps)

    if name == "muonh":
        # Full Delphi hybrid: MuonH (hidden) + AdamH (head) + AdamW (embed/scalar).
        # Single-device only: Newton-Schulz needs the full weight matrix, so this
        # raises under multi-rank sharding. The scaling sweep runs 1 GPU per job.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                raise RuntimeError(
                    "MuonH (optimizer.name='muonh') is single-device only; Newton-Schulz "
                    "orthogonalization requires the full weight matrix. Use 'adamh' for "
                    "multi-GPU/FSDP runs until the distributed MuonH variant lands."
                )
        from umich_slm.optim.muon import Muon, delphi_hybrid_param_groups

        groups = delphi_hybrid_param_groups(wrapper.inner)
        for g in groups:
            g["lr"] = cfg.lr
        return Muon(groups, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps)

    if name == "muonw":
        # Moonlight Muon (0.2*sqrt(max) update scale, decoupled decay) on hidden
        # Linear weights; AdamW, grouped exactly as the AdamW cells are, on the
        # rest. lr -> Muon group, lr_adam -> AdamW groups (None = lr). Under DDP
        # Newton-Schulz is sharded across ranks and all-gathered (muon_ns_shard),
        # which keeps the replicas bitwise identical.
        if float(getattr(cfg, "register_lr_mult", 1.0) or 1.0) != 1.0 or bool(
            getattr(cfg, "mat_ffn_freq_decay", False)
        ):
            raise ValueError(
                "muonw does not implement register_lr_mult or mat_ffn_freq_decay; "
                "both are AdamW-path instruments"
            )
        from umich_slm.optim.muonw import MuonW, muonw_param_groups

        groups = muonw_param_groups(
            wrapper.inner,
            wrapper.param_groups(weight_decay=cfg.weight_decay),
            lr=cfg.lr,
            lr_adam=cfg.lr_adam if cfg.lr_adam is not None else cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        n_muon = sum(p.numel() for p in groups[0]["params"])
        n_all = sum(p.numel() for g in groups for p in g["params"])
        _print_rank0(
            f"[titan] muonw: {len(groups[0]['params'])} matrices / {n_muon:,} params on Muon "
            f"({100.0 * n_muon / max(n_all, 1):.1f}%), rest on AdamW"
        )
        return MuonW(
            groups,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=tuple(cfg.betas),
            eps=cfg.eps,
            ns_shard=bool(cfg.muon_ns_shard),
        )

    groups = wrapper.param_groups(weight_decay=cfg.weight_decay)
    reg_mult = float(getattr(cfg, "register_lr_mult", 1.0) or 1.0)
    if reg_mult != 1.0:
        # swa_registers LR-calibration instrument: split the register bank and
        # its read gates into their own groups at lr * mult. Zero-init gates
        # bottleneck the bank's gradient early on, so the multiplier tests
        # whether the read needs a hotter LR to open within a sweep horizon.
        # Decay convention matches wrapper.param_groups (bank 2-D -> decay,
        # gates 1-D -> no decay). AdamW path only.
        def _is_reg(n: str) -> bool:
            return "register_tokens" in n or "register_gate" in n

        reg_decay, reg_no_decay, decay, no_decay = [], [], [], []
        for n, p in wrapper.named_parameters():
            if not p.requires_grad:
                continue
            nd = p.ndim <= 1 or n.endswith(".bias") or "norm" in n.lower()
            bucket = (
                (reg_no_decay if nd else reg_decay) if _is_reg(n) else (no_decay if nd else decay)
            )
            bucket.append(p)
        groups = [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
            {"params": reg_decay, "weight_decay": cfg.weight_decay, "lr": cfg.lr * reg_mult},
            {"params": reg_no_decay, "weight_decay": 0.0, "lr": cfg.lr * reg_mult},
        ]
        groups = [g for g in groups if g["params"]]
    if bool(getattr(cfg, "mat_ffn_freq_decay", False)):
        # The nested FFN's tiers are SLICES of single tensors, so they cannot be
        # split across param groups. Hand their decay to
        # apply_frequency_scaled_decay (called after optimizer.step()) and zero
        # it here, so each row is decayed in proportion to how often it is
        # actually trained -- 100/40/18/6% across the four tiers.
        from umich_slm.utils.mat_ffn_decay import mat_ffn_wd_exempt_ids

        exempt = mat_ffn_wd_exempt_ids(wrapper.inner)
        if exempt:
            split = []
            for g in groups:
                keep = [p for p in g["params"] if id(p) not in exempt]
                move = [p for p in g["params"] if id(p) in exempt]
                if keep:
                    split.append({**g, "params": keep})
                if move:
                    split.append({**g, "params": move, "weight_decay": 0.0})
            groups = split
            _print_rank0(
                f"[titan] mat_ffn frequency-scaled decay: {len(exempt)} tensors "
                f"moved to weight_decay=0 (decay applied post-step instead)"
            )
    if split_borrowed:
        groups = _split_borrowed_groups(groups, wrapper)
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
    for name in (
        "job",
        "model",
        "data",
        "training",
        "optimizer",
        "lr_scheduler",
        "parallelism",
        "logging",
        "checkpoint",
    ):
        section = getattr(config, name, None)
        if section is not None:
            serialized[name] = asdict(section)
    serialized["_hw"] = hw_info
    try:
        run = wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=config.job.run_name,
            config=serialized,
            tags=config.logging.wandb_tags or None,
            dir=config.job.output_dir,
            reinit=True,
        )
    except Exception as exc:  # noqa: BLE001 -- logging must never kill a run
        # A logging backend is not a training dependency: metrics.jsonl carries
        # everything the sweep reads. Letting wandb.init() raise is actively
        # dangerous under DDP -- rank 0 dies here while the other ranks sit in
        # the next collective until the 600s NCCL watchdog aborts them, so one
        # missing API key burns the whole allocation. That is exactly what took
        # out all 6 cells of Great Lakes job 58822458 and all 8 live-training
        # cells of 58822457 on 2026-08-26 ("No API key configured").
        print(f"[titan] wandb.init failed ({exc}); metrics.jsonl only", flush=True)
        return None
    return run


# ------------------------------------------------------------------- helpers
def _resolve_arrow_path(dataset_dir: str, leaf: str) -> Path:
    p = Path(leaf)
    if p.is_absolute():
        return p
    return Path(dataset_dir) / leaf


def _resolve_sft_shards(patterns, label: str) -> List[Path]:
    """Expand [sft].train_shards / val_shards globs into a sorted shard list.

    Hard-errors on an empty expansion. A missing glob would otherwise reach
    PackedSFTDataset, whose readers raise only once a reader produces no window
    -- i.e. after model build, compile and the first step, deep into a job.
    """
    import glob as _glob

    out: List[Path] = []
    for pat in patterns:
        # glob.glob, not Path.glob: it takes absolute patterns directly, which
        # every real [sft] shard list uses.
        hits = [Path(h) for h in sorted(_glob.glob(pat, recursive=True))]
        if not hits:
            raise ValueError(f"[sft].{label}_shards pattern matched nothing: {pat!r}")
        out.extend(hits)
    # Sorted + de-duplicated so every rank derives the identical ordering; the
    # reader stride is an index into THIS list and must agree across ranks.
    uniq = sorted({p.resolve() for p in out})
    if not uniq:
        raise ValueError(f"[sft].{label}_shards expanded to nothing")
    return uniq


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
_COMPILE_HEALTH_STEP = 50  # far enough in that compile has settled, early enough to act on


def run(config: TitanConfig, *, rank: int, local_rank: int, world_size: int) -> None:
    output_dir = Path(config.job.output_dir)
    if is_rank_zero():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Idempotency: a completed run leaves summary.json. Re-running the same
    # config is a no-op — exit before building the model. Every rank sees the
    # same file and returns together, before any collective op, so this is
    # safe under DDP.
    if (output_dir / "summary.json").exists():
        _print_rank0(
            f"[titan] summary.json present in {output_dir}; run already complete — skipping"
        )
        return

    torch.manual_seed(config.job.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.job.seed + rank)
        device = torch.device(f"cuda:{local_rank}")
        # Allow TF32 for any residual fp32 matmul (the loss/LM-head path; the bulk
        # of matmuls are already bf16 under autocast). Free throughput, negligible
        # numeric effect (< seed noise), applied uniformly to all runs from here on.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")

    # Tokenizer first — it determines the model's vocab_size.
    tokenizer = _load_tokenizer(config)
    vocab_size = len(tokenizer)
    if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size:
        # Use the larger of len() (covers added tokens) and vocab_size.
        vocab_size = max(vocab_size, int(tokenizer.vocab_size))
    _print_rank0(f"[titan] tokenizer={tokenizer.__class__.__name__} vocab_size={vocab_size}")

    capability_tracker = None
    if config.eval.capability_eval_set:
        try:
            from umich_slm.eval.tracking import CapabilityTracker

            capability_tracker = CapabilityTracker.from_eval_set(
                config.eval.capability_eval_set, tokenizer, device=str(device)
            )
            _print_rank0(f"[titan] {capability_tracker.describe()}")
        except Exception as exc:
            # A missing or stale eval set must not stop a training run.
            _print_rank0(f"[titan] capability tracking disabled: {type(exc).__name__}: {exc}")

    # Per-token loss signals ([token_loss]): Harmony annotations emitted by the
    # loaders, consumed by modules/agentic_heads.py. All off by default, and then
    # neither the loaders nor the forward change at all.
    tl_cfg = config.token_loss
    annotate_opts: Optional[AnnotateOptions] = None
    if tl_cfg.enabled:
        annotate_opts = AnnotateOptions(
            enabled=True,
            w_tool=float(tl_cfg.w_tool),
            errhead=float(tl_cfg.errhead_weight) > 0,
            unlik=float(tl_cfg.unlik_weight) > 0,
            unlik_mode=str(tl_cfg.unlik_mode),
        )
        if annotate_opts.errhead:
            # The head is a model submodule so it checkpoints with the weights
            # (and can serve as a verifier later); use_errhead rides in config.json.
            config.model.overrides = {**(config.model.overrides or {}), "use_errhead": True}
        _print_rank0(
            f"[titan] token_loss: w_tool={tl_cfg.w_tool} errhead_weight={tl_cfg.errhead_weight} "
            f"unlik_weight={tl_cfg.unlik_weight} unlik_mode={tl_cfg.unlik_mode} "
            f"(role metrics on)"
        )

    # Model.
    wrapper = build_wrapper_from_config(
        config.model, vocab_size=vocab_size, seq_len=config.data.seq_len
    )
    wrapper = wrapper.to(device)
    # After .to(): _apply copies each buffer independently, so sharing earlier
    # would be undone. Frees ~2.7 GB at deep80/32K (see share_rope_caches).
    wrapper.share_rope_caches()

    if config.training.gradient_checkpointing:
        wrapper.apply_gradient_checkpointing()
    # Order matters: float8 must run before compile so the captured graph
    # reflects the fp8 GEMMs. Float8 also requires Hopper+, so apply_float8
    # raises a clear error on older GPUs.
    if config.training.float8:
        recipe = getattr(config.training, "float8_recipe", None)
        converted = wrapper.apply_float8(recipe=recipe)
        _print_rank0(
            f"[titan] float8 ({recipe or 'torchao default'}) applied to "
            f"{converted} linear layers"
        )
    if config.training.torch_compile:
        # Dynamo's DDPOptimizer splits the compiled graph at gradient-bucket
        # boundaries to overlap all-reduce with compute. That split changes
        # AOTAutograd's saved-tensor partition between the forward and the
        # gradient-checkpoint recompute, so the window-LSE arms (sink/scribe are
        # the only ones that call flash_attn with return_attn_probs=True) hit
        # CheckpointError ("A different number of tensors was saved during the
        # original forward and recomputation", 42 vs 40) on the FIRST backward
        # under DDP+compile. Single-GPU is unaffected (no DDPOptimizer), which is
        # why it slips past 1-GPU smoke checks. Disabling the split keeps compile
        # and DDP correctness; the only cost is losing comm/compute overlap.
        # Scoped to the arms that need it, verified on the LIVE modules rather
        # than sniffed from config (CLAUDE.md: confirm the knob reached the
        # model). Validated on 2xA6000: control CheckpointError -> PASS.
        # dist.is_initialized(), not world_size > 1: the TITAN_DDP_SINGLE_RANK bench
        # path is DDP at world 1, and it is where the split/recompute mismatch surfaced.
        if dist.is_initialized() and config.training.gradient_checkpointing:
            # NB: alias the import — a bare `import torch._dynamo` binds the name
            # `torch` as a function-local, shadowing the module-level torch for
            # the whole function (UnboundLocalError at the earlier torch.* uses).
            import torch._dynamo as _dynamo

            wants_lse = any(getattr(m, "_wants_window_lse", False) for m in wrapper.modules())
            if not config.training.ddp_compile_split and _dynamo.config.optimize_ddp is not False:
                _dynamo.config.optimize_ddp = False
                _print_rank0("[titan] optimize_ddp=False (training.ddp_compile_split=false)")
            if wants_lse and _dynamo.config.optimize_ddp is not False:
                _dynamo.config.optimize_ddp = False
                _print_rank0(
                    "[titan] optimize_ddp=False: window-LSE (sink/scribe) arms "
                    "desync grad-checkpoint recompute under the DDPOptimizer "
                    "graph-split -> CheckpointError; disabling it keeps compile."
                )
        compiled = wrapper.apply_compile(
            deltanet_compile_safe=config.training.deltanet_compile_safe,
            window_lse_compile_safe=config.training.window_lse_compile_safe,
            safe_pointwise_autotune=config.training.safe_pointwise_autotune,
            compile_dyncompute_blocks=config.training.compile_dyncompute_blocks,
        )
        _print_rank0(f"[titan] torch.compile applied to {compiled} decoder blocks")
    compiled_blocks = compiled if config.training.torch_compile else 0

    # Numerical precision: mixed bf16 via autocast (no GradScaler needed).
    autocast_dtype = torch.bfloat16 if config.training.bf16 else torch.float32
    use_autocast = config.training.bf16 and device.type == "cuda"

    n_params = wrapper.num_params
    n_active = wrapper.num_active_params
    _print_rank0(
        f"[titan] params total={n_params/1e6:.2f}M active={n_active/1e6:.2f}M "
        f"size={config.model.size}"
    )

    dp_mode = str(config.parallelism.data_parallel).lower()
    if dp_mode == "fsdp":
        if str(config.optimizer.name).lower() not in ("adamw", "adamw_fused"):
            raise ValueError(
                f"parallelism.data_parallel='fsdp' supports AdamW only; got "
                f"optimizer.name={config.optimizer.name!r} (Muon variants need full matrices)"
            )
        model = apply_fsdp(
            wrapper,
            wrapper._iter_decoder_layers(),
            param_dtype=str(config.parallelism.fsdp_param_dtype),
            reshard_after_forward=bool(config.parallelism.fsdp_reshard_after_forward),
        )
        _print_rank0(
            f"[titan] FSDP2 over {world_size} rank(s): params/grads/optimizer sharded, "
            f"param_dtype={config.parallelism.fsdp_param_dtype}, "
            f"reshard_after_forward={config.parallelism.fsdp_reshard_after_forward}"
        )
    elif dp_mode == "ddp":
        model = apply_ddp(
            wrapper,
            local_rank=local_rank,
            find_unused_parameters=bool(config.training.ddp_find_unused_parameters),
            static_graph=bool(config.training.ddp_static_graph),
            comm_hook=str(config.training.ddp_comm_hook),
        )
    else:
        raise ValueError(f"parallelism.data_parallel must be 'ddp' or 'fsdp'; got {dp_mode!r}")

    # Optimizer + scheduler.
    b_mult = float(getattr(config.training, "borrowed_lr_mult", 1.0) or 1.0)
    b_hold = int(getattr(config.training, "borrowed_lr_hold_steps", 0) or 0)
    b_ramp = int(getattr(config.training, "borrowed_lr_ramp_steps", 0) or 0)
    split_borrowed = b_mult != 1.0
    if split_borrowed and int(getattr(config.training, "freeze_body_steps", 0) or 0) > 0:
        raise ValueError("borrowed_lr_mult and freeze_body_steps are alternatives; set one")
    if split_borrowed and not 0.0 < b_mult < 1.0:
        raise ValueError(f"training.borrowed_lr_mult must be in (0, 1); got {b_mult}")
    optimizer = _make_optimizer(wrapper, config.optimizer, split_borrowed=split_borrowed)
    lr_lambda = _make_lr_lambda(config.lr_scheduler)
    if split_borrowed:
        # One lambda per group: the borrowed body rides the same schedule times
        # its ramp, so it shares the warmup and the decay with everything else.
        def _borrowed_lambda(step: int) -> float:
            return lr_lambda(step) * _borrowed_lr_mult(step, b_mult, b_hold, b_ramp)

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=[
                _borrowed_lambda if g.get("borrowed") else lr_lambda for g in optimizer.param_groups
            ],
        )
        n_b = sum(
            p.numel() for g in optimizer.param_groups if g.get("borrowed") for p in g["params"]
        )
        n_all = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
        _print_rank0(
            f"[titan] borrowed-LR ramp: {n_b/1e6:.2f}M of {n_all/1e6:.2f}M params "
            f"({100.0 * n_b / max(n_all, 1):.1f}%) at {b_mult:g}x the schedule for steps "
            f"0-{b_hold - 1}, log-linear to 1x by step {b_hold + b_ramp}"
        )
    else:
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    _borrowed_gi = next(
        (i for i, g in enumerate(optimizer.param_groups) if g.get("borrowed")), None
    )

    # Data. "lm" reads the Arrow text column; "sft" reads Harmony jsonl.gz
    # shards and trains only on assistant spans.
    is_sft = str(config.data.kind).lower() == "sft"
    train_arrow = val_arrow = None
    if not is_sft:
        train_arrow = _resolve_arrow_path(config.data.dataset_dir, config.data.train_arrow)
        val_arrow = _resolve_arrow_path(config.data.dataset_dir, config.data.val_arrow)
    # Replay pool (train only; validation stays on the primary pool's held-out split).
    replay_frac = float(config.data.replay_frac)
    replay_arrow = None
    if replay_frac > 0:
        replay_arrow = _resolve_arrow_path(config.data.replay_dataset_dir, config.data.replay_arrow)
        if not replay_arrow.exists():
            raise FileNotFoundError(f"[data] replay arrow not found: {replay_arrow}")
        if rank == 0:
            print(f"[titan] replay: {replay_frac:.0%} of train windows from {replay_arrow}",
                  flush=True)

    # Intra-document masking (W2): when the model enables it (model.overrides
    # use_intra_doc_masking=true), the loader must emit per-token document_ids and the step
    # must forward them so the model can block cross-doc attention + reset RoPE per document.
    intra_doc_masking = bool(getattr(unwrap_ddp(model).config, "use_intra_doc_masking", False))

    if is_sft:
        # Checked against the LIVE model, not the TOML: an SFT window packs
        # several independent conversations, so without masking they attend
        # across each other and nothing raises. The packing is only sound
        # because document_ids separate them.
        if not intra_doc_masking:
            raise ValueError(
                "[data].kind = 'sft' requires use_intra_doc_masking on the model "
                "(set model.overrides.use_intra_doc_masking = true). Packed "
                "conversations would otherwise attend across each other silently."
            )
        sft_train_shards = _resolve_sft_shards(config.sft.train_shards, "train")
        sft_val_shards = (
            _resolve_sft_shards(config.sft.val_shards, "val") if config.sft.val_shards else []
        )
        if rank == 0:
            print(
                f"[data] sft: {len(sft_train_shards)} train shard(s), "
                f"{len(sft_val_shards)} val shard(s), "
                f"long_policy={config.sft.long_policy}",
                flush=True,
            )

    # Token-Superposition Training (TST). When enabled, the loader runs at
    # ``seq_len * bag_size`` for the whole stage to preserve the paper's
    # equal-FLOPs-per-step contract; phase (ii) slices batches back to seq_len.
    tst_cfg = config.training.tst
    tst_extend = bool(tst_cfg.enabled and tst_cfg.extend_seq_len_for_equal_flops)
    loader_seq_len = config.data.seq_len * tst_cfg.bag_size if tst_extend else config.data.seq_len

    def make_train_loader():
        if is_sft:
            return build_sft_dataloader(
                sft_train_shards,
                tokenizer,
                seq_len=loader_seq_len,
                batch_size=config.training.per_device_batch_size,
                rank=rank,
                world_size=world_size,
                seed=config.job.seed,
                num_workers=config.data.num_workers,
                infinite=True,
                long_policy=config.sft.long_policy,
                source_epochs=config.sft.source_epochs,
                shuffle_buffer=config.data.shuffle_buffer,
                annotate=annotate_opts,
            )
        return build_dataloader(
            train_arrow,
            tokenizer,
            seq_len=loader_seq_len,
            batch_size=config.training.per_device_batch_size,
            text_column=config.data.text_column,
            rank=rank,
            world_size=world_size,
            seed=config.job.seed,
            num_workers=config.data.num_workers,
            infinite=True,
            emit_doc_ids=intra_doc_masking,
            replay_arrow_path=replay_arrow,
            replay_frac=replay_frac,
            annotate=annotate_opts,
        )

    def make_val_loader():
        # infinite=False + a fixed seed that is NOT job.seed: every run, every
        # architecture and every optimizer arm must see the SAME held-out windows
        # in the same order, or the losses we rank on are not comparable.
        if is_sft:
            return build_sft_dataloader(
                sft_val_shards,
                tokenizer,
                seq_len=config.data.seq_len,
                batch_size=config.training.per_device_batch_size,
                rank=rank,
                world_size=world_size,
                seed=config.eval.seed,
                num_workers=config.data.num_workers,
                infinite=False,
                long_policy=config.sft.long_policy,
                # No source_epochs and no shuffle on the held-out set: the eval
                # windows must be identical across arms and across phases, or
                # the losses being compared are not the same measurement.
                source_epochs=None,
                shuffle_buffer=0,
                annotate=annotate_opts,
            )
        return build_dataloader(
            val_arrow,
            tokenizer,
            seq_len=config.data.seq_len,
            batch_size=config.training.per_device_batch_size,
            text_column=config.data.text_column,
            rank=rank,
            world_size=world_size,
            seed=config.eval.seed,
            num_workers=config.data.num_workers,
            infinite=False,
            emit_doc_ids=intra_doc_masking,
            annotate=annotate_opts,
        )

    @torch.no_grad()
    def evaluate() -> Dict[str, float]:
        """Mean held-out loss over a fixed number of validation batches.

        Reports ``lm_loss`` separately from ``loss``: for MoD/MTP architectures the
        fused ``loss`` carries an aux/MTP term that is not a language-modelling
        quantity, so ranking on it would compare architectures on different
        objectives. The sweep ranks on ``val/lm_loss``.
        """
        was_training = model.training
        model.eval()
        totals: Dict[str, float] = {"loss": 0.0, "lm_loss": 0.0}
        counts: Dict[str, int] = {"loss": 0, "lm_loss": 0}
        sft_loss_sum = 0.0
        sft_tok_sum = 0
        tok_stats = TokenStatAccumulator(collect_scores=True)
        bs = max(int(config.training.per_device_batch_size), 1)
        n_batches = max(math.ceil(int(config.eval.sequences) / (bs * max(world_size, 1))), 1)
        seen = 0
        for batch in make_val_loader():
            if seen >= n_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            document_ids = (
                batch["document_ids"].to(device, non_blocking=True)
                if intra_doc_masking and "document_ids" in batch
                else None
            )
            ctx = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if use_autocast
                else nullcontext()
            )
            tt = (
                token_targets_from_batch(batch, device, tl_cfg) if annotate_opts is not None else None
            )
            with ctx:
                output = model(
                    input_ids,
                    labels=labels,
                    document_ids=document_ids,
                    **({"token_targets": tt} if tt else {}),
                )
            comps = loss_components(output)
            tok_stats.add(output.token_stats, batch.get("errhead_labels"))
            if is_sft:
                # labels[:, 1:] is the model's own denominator -- position 0 is
                # never a target -- and sft_collate computed it in the worker.
                n_tok = int(batch.get("n_loss_tokens", 0))
                per_tok = comps.get("lm_loss", comps.get("loss"))
                if per_tok is not None and n_tok > 0:
                    sft_loss_sum += float(per_tok) * n_tok
                    sft_tok_sum += n_tok
            for k in totals:
                v = comps.get(k)
                if v is not None:
                    totals[k] += v
                    counts[k] += 1
            seen += 1
        if was_training:
            model.train()
        # asst_loss / errhead_auroc / errhead_loss / unlik_loss (token-weighted,
        # all-reduced). A collective, so every rank must reach it.
        tok_out = tok_stats.result(world_size, device) if annotate_opts is not None else {}

        if is_sft:
            # Token-weighted, all-reduced, and written to its OWN key.
            #
            # Token-weighted because the model returns sum(CE)/count(valid) for
            # its own batch and the loss-bearing fraction swings several-fold
            # between windows under a mask -- a plain mean over batches would
            # weight a sparse-loss agentic window the same as a dense one.
            #
            # All-reduced because with DDP each rank sees a different slice;
            # rank 0's local mean is not the held-out loss.
            #
            # And "sft_loss", never "lm_loss": a masked loss sits several nats
            # below an unmasked FineWeb loss, and the sweep ranks arms on
            # val_lm_loss -- an SFT run written into that key would rank as the
            # best pretraining arm in the grid. val_lm_loss stays absent, hence
            # null in summary.json.
            t = torch.tensor([sft_loss_sum, float(sft_tok_sum)], dtype=torch.float64, device=device)
            if world_size > 1 and dist.is_initialized():
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            loss_sum, tok_sum = float(t[0].item()), float(t[1].item())
            out = {}
            if tok_sum > 0:
                out["sft_loss"] = loss_sum / tok_sum
            out["loss_tokens"] = tok_sum
            out["sequences"] = float(seen * bs * max(world_size, 1))
            out.update(tok_out)
            return out

        out = {k: (totals[k] / counts[k]) for k in totals if counts[k]}
        # lm_loss is None on architectures that never populate it; fall back so the
        # ranking metric always exists.
        if "lm_loss" not in out and "loss" in out:
            out["lm_loss"] = out["loss"]
        out["sequences"] = float(seen * bs * max(world_size, 1))
        out.update(tok_out)
        return out

    train_loader = make_train_loader()
    train_iter = iter(train_loader)

    # Resume. Explicit resume_from wins; otherwise auto-detect the newest
    # checkpoint-<step> in output_dir so a re-submitted partial run continues
    # instead of restarting (idempotent under wall-clock kills). If the newest
    # was written half-way through a kill, fall back to the next-older one.
    start_step = 0
    resumed = False
    if config.checkpoint.resume_from:
        candidates = [Path(config.checkpoint.resume_from)]
    else:
        found = [
            p
            for p in output_dir.glob("checkpoint-*")
            if p.is_dir() and p.name.split("-")[-1].isdigit()
        ]
        candidates = sorted(found, key=lambda p: int(p.name.split("-")[-1]), reverse=True)
        if candidates:
            _print_rank0(
                f"[titan] auto-resume: found {len(candidates)} checkpoint(s), newest {candidates[0]}"
            )
    if dp_mode == "fsdp" and (
        candidates or config.checkpoint.init_weights_from or config.checkpoint.grow_from
    ):
        # Outside the try below on purpose: that loop treats a failed load as a
        # corrupt checkpoint and falls back to older ones, then to step 0, so an
        # FSDP load that cannot work would silently restart the run.
        raise NotImplementedError(
            "FSDP cannot resume or init-load yet (sharded params need "
            "set_model_state_dict); resume this run under DDP, which reads the full "
            f"checkpoints FSDP writes. Found: {candidates[:1] or 'init/grow weights'}"
        )
    for ckpt in candidates:
        try:
            start_step = load_checkpoint(
                ckpt,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
            )
            _print_rank0(f"[titan] resumed from {ckpt} at step {start_step}")
            resumed = True
            break
        except Exception as exc:  # noqa: BLE001 — corrupt/partial checkpoint, try older
            _print_rank0(f"[titan] checkpoint {ckpt} unusable ({exc}); trying older")
            start_step = 0

    # Weights-only init for a NEW run (e.g. a context-extension stage). Only when
    # nothing of this run's own was resumed: a requeued extension job must pick up
    # its own checkpoint-*, not restart from the parent's weights. Deliberately NOT
    # wrapped in try/except -- a config mismatch or a missing key here means the run
    # would silently train a random or different model, so it must abort.
    if not resumed and config.checkpoint.init_weights_from:
        changed = load_init_weights(
            Path(config.checkpoint.init_weights_from),
            model,
            allow_config_diff=config.checkpoint.init_weights_allow_config_diff,
            # The P3 failed-call head is an auxiliary head: a run may add it on
            # top of a checkpoint trained without it (fresh zero-init) or drop it.
            optional_prefixes=("errhead.",),
        )
        _print_rank0(
            f"[titan] initialised weights from {config.checkpoint.init_weights_from} "
            f"(fresh optimizer, step 0); config changes vs checkpoint: "
            + (", ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in changed.items()) or "none")
        )
    if not resumed and config.checkpoint.grow_from:
        if config.checkpoint.init_weights_from:
            raise ValueError("checkpoint.grow_from and checkpoint.init_weights_from are exclusive")
        from umich_slm.warmstart.grow import load_grown_weights

        report = load_grown_weights(
            Path(config.checkpoint.grow_from),
            model,
            operator=config.checkpoint.grow_operator,
            shrink=config.checkpoint.grow_shrink,
            allow_config_diff=config.checkpoint.init_weights_allow_config_diff,
        )
        _print_rank0(
            f"[titan] grew weights from {config.checkpoint.grow_from} (fresh optimizer, "
            f"step 0): {json.dumps(report, default=str)}"
        )

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
        world_size * config.training.per_device_batch_size * grad_accum * config.data.seq_len
    )
    log_history: List[Dict[str, Any]] = []
    model.train()

    # TST controller: drives the phase (i) → phase (ii) cutover within this stage.
    tst_controller = (
        TSTController(
            enabled=bool(tst_cfg.enabled),
            bag_size=int(tst_cfg.bag_size),
            recovery_ratio=float(tst_cfg.recovery_ratio),
        )
        if tst_cfg.enabled
        else None
    )
    if tst_controller is not None:
        # Propagate loss weighting to the model config so the forward path uses it.
        unwrap_ddp(model).config.tst_loss_weighting = tst_cfg.loss_weighting
        tst_controller.begin(config.training.max_steps, unwrap_ddp(model))

    # mat_ffn nested-dropout warmup: for the first `mat_ffn_warmup_frac` of the
    # run the tier assignment is SAMPLED from a prior annealing uniform -> the
    # arm's target skew, so every nested width is trained from step 0 and an
    # untrained router cannot starve the widest tier. No-op on every arm that
    # does not set the flag. See src/umich_slm/callbacks/mat_ffn_warmup.py.
    _mcfg = unwrap_ddp(model).config
    _mat_warmup_frac = float(getattr(_mcfg, "mat_ffn_warmup_frac", 0.0) or 0.0)
    mat_ffn_warmup = MatFFNWarmupController(
        enabled=bool(getattr(_mcfg, "use_mat_ffn", False)) and _mat_warmup_frac > 0.0,
        warmup_frac=_mat_warmup_frac,
        stages=int(getattr(_mcfg, "mat_ffn_warmup_stages", 4) or 4),
        verbose=(rank == 0),
    )
    mat_ffn_warmup.begin(config.training.max_steps, unwrap_ddp(model))

    _print_rank0(
        f"[titan] starting training "
        f"max_steps={config.training.max_steps} "
        f"tokens_per_step={tokens_per_step:,} "
        f"world_size={world_size} bf16={config.training.bf16}"
    )

    _mat_ffn_interval = int(getattr(config.training, "mat_ffn_stats_interval", 0) or 0)

    def _mat_ffn_stats(step_idx: int) -> dict:
        # Drained on a cadence: the counters accumulate every step for free, but
        # reading them is a host sync, so amortize it over mat_ffn_stats_interval.
        if _mat_ffn_interval <= 0 or (step_idx + 1) % _mat_ffn_interval != 0:
            return {}
        try:
            payload = mat_ffn_occupancy_payload(model)
        except Exception as exc:  # telemetry must never take down a run
            _print_rank0(f"[titan] mat_ffn occupancy telemetry failed: {exc}")
            return {}
        # Loud, not just logged: both signatures of the 2026-09-18 failure, where
        # every threshold sat at 0 and the budget clip granted width by position
        # for whole runs while occupancy read exactly on target.
        floor = payload.get("mat_ffn/theta_floor_frac")
        ratio = payload.get("mat_ffn/width_half_ratio")
        if floor is not None and floor >= 0.99:
            _print_rank0(
                f"[titan] WARNING step {step_idx + 1}: mat_ffn thresholds at the 0 floor "
                f"({floor:.0%} of layer x tier) -- routing is by POSITION, not content"
            )
        if ratio is not None and ratio > 1.5:
            _print_rank0(
                f"[titan] WARNING step {step_idx + 1}: mat_ffn first-half mean width is "
                f"{ratio:.2f}x the second half -- routing looks positional"
            )
        return payload

    # Grad-spike guard: a step whose pre-clip norm is non-finite or >> an EMA of
    # healthy norms has its optimizer update skipped. A sustained STREAK of skips
    # re-seeds the EMA instead of wedging the run -- see titan/grad_guard.py.
    _freq_decay_wd = (
        float(config.optimizer.weight_decay)
        if bool(getattr(config.optimizer, "mat_ffn_freq_decay", False))
        else 0.0
    )

    grad_guard = GradSpikeGuard(
        enabled=config.training.grad_spike_skip,
        factor=config.training.grad_spike_factor,
        warmup=config.training.grad_spike_warmup,
        recover_after=config.training.grad_spike_recover_after,
        max_rewarms=config.training.grad_spike_max_rewarms,
    )

    # Pause-stream phasing setup (modules/pause_stream.py). Resolved once, before
    # the loop, so the per-step check is a plain int compare.
    time_optimizer = bool(config.logging.time_optimizer) and torch.cuda.is_available()
    optim_ms = None
    _pause_frac = float(getattr(config.training, "pause_phase_start_frac", 0.0))
    _raw_model = unwrap_ddp(model)
    _has_pause = _raw_model.has_pause_stream
    if _pause_frac > 0.0 and not _has_pause:
        raise ValueError(
            f"training.pause_phase_start_frac={_pause_frac} but {config.model.size} "
            "has no pause stream; the knob would do nothing"
        )
    pause_phase_steps = int(_pause_frac * config.training.max_steps) if _pause_frac > 0.0 else None
    _mtp_frac = float(getattr(config.training, "mtp_phase_start_frac", 0.0))
    if _mtp_frac > 0.0 and not _raw_model.has_mtp:
        raise ValueError(
            f"training.mtp_phase_start_frac={_mtp_frac} but {config.model.size} "
            "has no MTP heads; the knob would do nothing"
        )
    mtp_phase_steps = int(_mtp_frac * config.training.max_steps) if _mtp_frac > 0.0 else None
    mtp_phase_on = True
    if mtp_phase_steps is not None:
        mtp_phase_on = _raw_model.set_mtp_enabled(False)
        _print_rank0(
            f"[mtp] multi-token prediction starts at step {mtp_phase_steps} "
            f"({_mtp_frac:.3g} of {config.training.max_steps})"
        )
    pause_phase_on = True
    if pause_phase_steps is not None:
        pause_phase_on = _raw_model.set_pause_enabled(False)
        _print_rank0(
            f"[pause] separation starts at step {pause_phase_steps} "
            f"({_pause_frac:.3g} of {config.training.max_steps})"
        )

    # Embedding healing: for the first freeze_body_steps steps, let ONLY the two
    # embedding tables move, so composed embeddings realign into the coordinate
    # system the transplanted body already expects instead of the body chasing
    # them. See docs/parameter_warm_start.md.
    #
    # This is done by DROPPING body gradients before the clip, NOT by toggling
    # requires_grad. Toggling looks cleaner and fails: jobs 64059/64069 both died
    # at the unfreeze step (1,525) with
    #   CheckpointError: Recomputed values ... have different metadata
    # and a saved/recomputed tensor list shifted by one. The decoder blocks are
    # torch.compiled AND gradient-checkpointed; dynamo does not re-trace on a
    # parameter's requires_grad flip, so after the unfreeze the checkpoint
    # recompute ran a graph traced under the frozen state and saved a different
    # number of tensors than the forward. Same failure class as the mat_ffn theta
    # write documented below (jobs 56494/59094314).
    #
    # p.grad = None leaves the graph, the compiled artifact and the saved-tensor
    # set identical at every step. torch optimizers skip a param whose .grad is
    # None, so there is no update AND no decoupled weight decay on the body. It
    # also works under DDP -- gradients are still produced and all-reduced, we
    # just discard them after -- so the old single-rank restriction is gone.
    # (mat_ffn_freq_decay would decay body rows outside the optimizer, but it is
    # an arm-specific flag and no warm-start arm sets it.)
    freeze_steps = int(getattr(config.training, "freeze_body_steps", 0) or 0)
    _body_params = [
        p_
        for name_, p_ in unwrap_ddp(model).named_parameters()
        if not any(k in name_ for k in _EMB_NAMES)
    ]
    _emb_count = sum(1 for _ in unwrap_ddp(model).named_parameters()) - len(_body_params)

    def _drop_body_grads() -> None:
        for p_ in _body_params:
            p_.grad = None

    if freeze_steps > 0:
        if start_step < freeze_steps:
            _print_rank0(
                f"[titan] embedding-healing: body held for steps "
                f"{start_step}-{freeze_steps - 1} "
                f"({len(_body_params)} body tensors dropped, {_emb_count} embedding "
                f"tensors trained); grad_norm below is EMBEDDING-ONLY until then"
            )
        else:
            _print_rank0(
                f"[titan] embedding-healing: resumed past the healing phase "
                f"(start_step={start_step} >= freeze_body_steps={freeze_steps}); body live"
            )

    for step in range(start_step, config.training.max_steps):
        healing = freeze_steps > 0 and step < freeze_steps
        if freeze_steps > 0 and step == freeze_steps:
            _print_rank0(
                f"[titan] embedding-healing: phase over at step {step}; body now trains"
            )
        timer.start()
        optimizer.zero_grad(set_to_none=True)
        # Every rank draws the same recurrent depths this step (and a resume
        # redraws them). No-op for arms that never loop. See recurrent_depth.py.
        seed_recursion_rng(config.job.seed, step)

        # TST phase tracking. After the cutover, slice each batch back to
        # config.data.seq_len so the recovery phase trains on standard-length
        # sequences (the loader produces seq_len * bag_size).
        tst_phase_active = (
            tst_controller.step(step, unwrap_ddp(model)) if tst_controller is not None else False
        )
        mat_ffn_warmup.step(step, unwrap_ddp(model))
        # Pause-stream phasing: train standard for the first fraction of the
        # schedule, then switch separation on for the rest. One recompile at the
        # switch; a no-op when the model has no pause stream.
        if pause_phase_steps is not None:
            want = step >= pause_phase_steps
            if want != pause_phase_on:
                pause_phase_on = unwrap_ddp(model).set_pause_enabled(want)
                _print_rank0(
                    f"[pause] state/prediction separation "
                    f"{'ON' if pause_phase_on else 'OFF'} at step {step}"
                )
        if mtp_phase_steps is not None:
            want = step >= mtp_phase_steps
            if want != mtp_phase_on:
                mtp_phase_on = unwrap_ddp(model).set_mtp_enabled(want)
                _print_rank0(f"[mtp] multi-token prediction {'ON' if mtp_phase_on else 'OFF'} at step {step}")
        # Threshold-router controller. Deliberately at the TOP of the step and
        # NOT inside forward: forward only measures the occupancy error, because
        # a checkpointed block's forward runs twice per step and a theta write in
        # there re-routed the recompute (CheckpointError, jobs 56494/59094314).
        # See swiglu.step_route_thresholds. No-op for every arm but mat_ffn_thresh*.
        step_route_thresholds(unwrap_ddp(model))

        # Compile-health check, once, after compile has had time to settle.
        # torch.compile runs with suppress_errors=True (deliberately -- a lazy
        # backend failure hours in would otherwise kill the run), so a frame it
        # abandons runs EAGER for the rest of the job and says nothing. That is
        # not hypothetical: the ar10m mat_ffn cell ran eager end to end and was
        # recorded at 1.79x dense when compiled it is 1.18x, i.e. a compile
        # artifact reached a walltime table as an ARCHITECTURE result. See
        # src/umich_slm/titan/compile_health.py.
        if step == start_step + _COMPILE_HEALTH_STEP and rank == 0:
            log_compile_health(compiled_blocks, step, log_fn=_print_rank0)

        # Only the logged steps need the scalar loss/components; accumulate the loss
        # on-device as a tensor and sync it (and the per-component floats) once, on
        # log steps, instead of a host .item() every micro-step (grad_accum syncs/
        # step). The grad-norm .item() below is unavoidable (the spike guard needs
        # it every step); this removes the others.
        is_log_step = ((step + 1) % max(config.logging.log_every, 1) == 0) or (step == start_step)
        accum_loss_t: Optional[torch.Tensor] = None
        component_accum: Dict[str, float] = {"lm_loss": 0.0, "mtp_loss": 0.0, "aux_loss": 0.0}
        component_counts: Dict[str, int] = {"lm_loss": 0, "mtp_loss": 0, "aux_loss": 0}
        step_tok_stats = TokenStatAccumulator() if annotate_opts is not None else None

        # Token-weighted gradient accumulation (SFT only).
        #
        # The model returns a mean over THIS microbatch's valid tokens, so under
        # a loss mask -- where the loss-bearing fraction swings several-fold
        # between windows -- dividing by grad_accum up-weights sparse-loss
        # agentic windows ~10x relative to their token share, i.e. optimises a
        # mixture nobody chose. The weights need every microbatch's token count
        # BEFORE the first backward, so the step's microbatches are fetched up
        # front (3 long tensors each: ~1.6 MB at B16 x 4096, negligible).
        #
        # DDP does NOT make this moot at grad_accum=1: its cross-rank gradient
        # mean has the identical flaw, which is why the denominator is reduced
        # across ranks rather than taken per-rank.
        micro_batches: List[dict] = []
        # Host time blocked on the loader. The previous step ended on the grad-norm
        # .item() sync, so the GPU is idle for all of it: this is pure stall.
        data_t0 = time.perf_counter()
        for _ in range(grad_accum):
            try:
                micro_batches.append(next(train_iter))
            except StopIteration:
                train_iter = iter(make_train_loader())
                micro_batches.append(next(train_iter))
        data_ms = (time.perf_counter() - data_t0) * 1000.0

        if is_sft:
            local_counts = [int(b.get("n_loss_tokens", 0)) for b in micro_batches]
            gtok = torch.tensor([float(sum(local_counts))], dtype=torch.float64, device=device)
            if world_size > 1 and dist.is_initialized():
                dist.all_reduce(gtok, op=dist.ReduceOp.SUM)
            global_tok = int(gtok[0].item())
            if global_tok <= 0:
                raise RuntimeError(
                    f"[titan] step {step + 1}: no loss-bearing tokens in the whole "
                    f"optimizer step (local counts {local_counts}). Every label is "
                    f"IGNORE_INDEX -- check the loss mask and long_policy."
                )
            micro_weights = accum_weights(local_counts, world_size, total=global_tok)
        else:
            micro_weights = [1.0 / grad_accum] * grad_accum

        for micro, batch in enumerate(micro_batches):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            document_ids = (
                batch["document_ids"].to(device, non_blocking=True)
                if intra_doc_masking and "document_ids" in batch
                else None
            )
            tt = None
            if annotate_opts is not None:
                tt = token_targets_from_batch(
                    batch,
                    device,
                    tl_cfg,
                    seq_len=config.data.seq_len if (tst_extend and not tst_phase_active) else None,
                )
            if tst_extend and not tst_phase_active:
                # Recovery: drop back to standard seq_len for the rest of the stage.
                input_ids = input_ids[:, : config.data.seq_len]
                labels = labels[:, : config.data.seq_len]
                if document_ids is not None:
                    document_ids = document_ids[:, : config.data.seq_len]

            no_sync = (
                model.no_sync()
                if hasattr(model, "no_sync") and micro < grad_accum - 1
                else nullcontext()
            )
            ctx = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if use_autocast
                else nullcontext()
            )
            with no_sync:
                with ctx:
                    output = model(
                        input_ids,
                        labels=labels,
                        document_ids=document_ids,
                        **({"token_targets": tt} if tt else {}),
                    )
                loss = output.loss * micro_weights[micro]
                loss.backward()
                ld = loss.detach()
                accum_loss_t = ld if accum_loss_t is None else accum_loss_t + ld

                if is_log_step:
                    comps = loss_components(output)
                    for k in component_accum:
                        v = comps.get(k)
                        if v is not None:
                            component_accum[k] += v
                            component_counts[k] += 1
                    if step_tok_stats is not None:
                        step_tok_stats.add(output.token_stats)

        if healing:
            # Before the clip, so the reported grad_norm is the embedding-only
            # norm and the guard's EMA tracks the phase actually being trained.
            _drop_body_grads()

        grad_norm = clip_grad_norm_(unwrap_ddp(model).parameters(), config.training.max_grad_norm)
        gn = (
            float(grad_norm.detach().float().item())
            if hasattr(grad_norm, "detach")
            else float(grad_norm)
        )
        # Grad-spike guard: skip the optimizer update on a non-finite or anomalously
        # large pre-clip grad norm so one poisoned batch cannot knock the model out
        # of its basin (which beta2=0.9999's stale second moment then makes
        # permanent). The LR schedule still advances (it is defined over loop steps),
        # so a skip just forfeits that batch's update.
        grad_skipped = grad_guard.check(gn, step - start_step)
        optim_ms = None
        if grad_skipped:
            optimizer.zero_grad(set_to_none=True)
            _print_rank0(
                f"[titan] step={step+1}: grad-spike guard SKIP "
                f"(grad_norm={gn:.3e}, ema={(grad_guard.ema or float('nan')):.3e}); "
                f"optimizer step forgone (total_skips={grad_guard.skips_total}, "
                f"consec={grad_guard.consec_skips})"
            )
            if grad_guard.last_event:
                _print_rank0(f"[titan] step={step+1}: grad-spike guard {grad_guard.last_event}")
        else:
            # The grad-norm .item() above already synced, so with time_optimizer
            # one trailing sync makes this the optimizer's own GPU time.
            optim_t0 = time.perf_counter()
            optimizer.step()
            if time_optimizer:
                torch.cuda.synchronize()
                optim_ms = (time.perf_counter() - optim_t0) * 1000.0
            grad_guard.applied(gn)
            if _freq_decay_wd > 0.0:
                # Decoupled decay is proportional to lr, so use the CURRENT
                # scheduled value -- a decayed schedule must decay the decay too.
                apply_frequency_scaled_decay(
                    model, lr=float(scheduler.get_last_lr()[0]), weight_decay=_freq_decay_wd
                )
        scheduler.step()

        elapsed_ms, peak_bytes = timer.end()
        elapsed_s = elapsed_ms / 1000.0
        tokens_per_sec = tokens_per_step / elapsed_s if elapsed_s > 0 else 0.0
        flops_total = flops_counter.per_step_flops(tokens_per_step=tokens_per_step)
        mfu_pct = mfu.mfu(total_flops=flops_total, elapsed_seconds=elapsed_s) * 100.0
        # HFU: what the silicon actually executes. Activation checkpointing
        # re-runs the forward in backward (4x fwd) while MFU credits 3x, so
        # with grad-ckpt on HFU is 4/3 of MFU. Log both — MFU stays comparable
        # to published numbers, HFU says how much headroom is really left.
        hw_flops = flops_counter.per_step_hardware_flops(
            tokens_per_step=tokens_per_step,
            gradient_checkpointing=bool(config.training.gradient_checkpointing),
        )
        hfu_pct = mfu.mfu(total_flops=hw_flops, elapsed_seconds=elapsed_s) * 100.0

        if is_log_step:
            # Single host sync for the logged loss (accum_loss_t already sums
            # output.loss/grad_accum across micro-steps, i.e. the per-micro mean).
            accum_loss = float(accum_loss_t.float().item()) if accum_loss_t is not None else 0.0
            avg_components = {
                k: (component_accum[k] / component_counts[k]) if component_counts[k] else None
                for k in component_accum
            }
            payload = {
                "step": step + 1,
                # accum_loss already sums (output.loss / grad_accum) across the
                # grad_accum micro-steps, i.e. it IS the per-micro-step mean loss.
                # (Do NOT multiply by grad_accum — that would report the *sum* and
                # inflate the logged loss by grad_accum×.)
                "loss": accum_loss,
                "grad_norm": gn,
                "grad_skipped": grad_skipped,
                "grad_skips_total": grad_guard.skips_total,
                "grad_consec_skips": grad_guard.consec_skips,
                "grad_rewarms": grad_guard.rewarms,
                "lr": float(scheduler.get_last_lr()[0]),
                **(
                    {"lr_adam": float(scheduler.get_last_lr()[1])}
                    if len(scheduler.get_last_lr()) > 1
                    else {}
                ),
                **(
                    {"lr_borrowed": float(scheduler.get_last_lr()[_borrowed_gi])}
                    if _borrowed_gi is not None
                    else {}
                ),
                "step_time_ms": elapsed_ms,
                "data_ms": data_ms,
                **({"optim_ms": optim_ms} if time_optimizer else {}),
                "tokens_per_sec": tokens_per_sec,
                "mfu_pct": mfu_pct,
                "hfu_pct": hfu_pct,
                "peak_mem_gb": peak_bytes / (1024**3),
                **{f"component/{k}": v for k, v in avg_components.items()},
                # Rank-local, like the components above: asst_loss, lm_obj (the
                # role-weighted LM term actually trained), errhead_loss, unlik_loss.
                **(
                    {f"component/{k}": v for k, v in step_tok_stats.result().items()}
                    if step_tok_stats is not None
                    else {}
                ),
                **gather_width_payload(),
                **_mat_ffn_stats(step),
            }
            log_history.append(payload)
            if is_rank_zero():
                jsonl.log(payload)
                if wandb_run is not None:
                    wandb_run.log(payload, step=step + 1)
                _print_rank0(
                    f"[titan] step={step+1}/{config.training.max_steps} "
                    f"loss={payload['loss']:.4f} lr={payload['lr']:.2e} "
                    f"tok/s={tokens_per_sec:,.0f} mfu={mfu_pct:.1f}% hfu={hfu_pct:.1f}% "
                    f"step_ms={elapsed_ms:.1f} mem={payload['peak_mem_gb']:.1f}GB"
                    + (
                        f" gather_widths={payload['mod/gather_widths_distinct']}"
                        if "mod/gather_widths_distinct" in payload
                        else ""
                    )
                )
                _churn = gather_width_warning(torch._dynamo.config.cache_size_limit)
                if _churn:
                    _print_rank0(f"[titan] {_churn}")

        # Periodic held-out eval.
        eval_every = max(int(config.eval.every), 0) if config.eval.enabled else 0
        if eval_every and (step + 1) % eval_every == 0:
            val = evaluate()
            payload = {"step": step + 1, **{f"val/{k}": v for k, v in val.items()}}
            # Per-domain capability metrics on the same fixed subset every time.
            # Deliberately run on EVERY rank rather than rank 0: the ranks are
            # identical DDP replicas so the work is redundant, but a rank-0-only
            # branch puts the peers into the next collective early and the NCCL
            # watchdog aborts the allocation (the failure mode that lost every
            # cell of 58822457).
            cap_every = max(int(config.eval.capability_every), 0) or eval_every
            if capability_tracker is not None and (step + 1) % cap_every == 0:
                try:
                    payload.update(
                        {
                            f"capability/{k}": v
                            for k, v in capability_tracker.evaluate(model).items()
                        }
                    )
                except Exception as exc:  # never let a metric kill a training run
                    _print_rank0(f"[titan] capability eval failed: {type(exc).__name__}: {exc}")
            log_history.append(payload)
            if is_rank_zero():
                jsonl.log(payload)
                if wandb_run is not None:
                    wandb_run.log(payload, step=step + 1)
                if is_sft:
                    _print_rank0(
                        f"[titan] eval step={step+1} "
                        f"val_sft_loss={val.get('sft_loss', float('nan')):.4f} "
                        f"({int(val.get('loss_tokens', 0)):,} loss tok, "
                        f"{int(val.get('sequences', 0))} seqs)"
                    )
                else:
                    _print_rank0(
                        f"[titan] eval step={step+1} "
                        f"val_loss={val.get('loss', float('nan')):.4f} "
                        f"val_lm_loss={val.get('lm_loss', float('nan')):.4f} "
                        f"({int(val.get('sequences', 0))} seqs)"
                    )
                if annotate_opts is not None:
                    _print_rank0(
                        "[titan] eval token_loss "
                        + " ".join(
                            f"{k}={val[k]:.4f}"
                            for k in ("asst_loss", "errhead_auroc", "errhead_loss", "unlik_loss")
                            if k in val
                        )
                        + f" calls={int(val.get('errhead_calls', 0))}"
                    )

        # Periodic checkpoint.
        save_every = max(int(config.checkpoint.save_every), 0)
        if save_every and (step + 1) % save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                step=step + 1,
                config=config,
                log_history=log_history,
            )
            barrier()

    # Final held-out eval. This is the number the scaling sweep ranks on, so it
    # runs unconditionally at the end of every run (independent of eval.every).
    final_val: Dict[str, float] = {}
    if config.eval.enabled:
        final_val = evaluate()
        payload = {
            "step": config.training.max_steps,
            **{f"val/{k}": v for k, v in final_val.items()},
            "final": True,
        }
        log_history.append(payload)
        if is_rank_zero():
            jsonl.log(payload)
            if wandb_run is not None:
                wandb_run.log(payload, step=config.training.max_steps)
            _print_rank0(
                f"[titan] FINAL eval "
                f"val_loss={final_val.get('loss', float('nan')):.4f} "
                + (
                    f"val_sft_loss={final_val.get('sft_loss', float('nan')):.4f} "
                    if is_sft
                    else f"val_lm_loss={final_val.get('lm_loss', float('nan')):.4f} "
                )
                + f"({int(final_val.get('sequences', 0))} seqs)"
            )
            if annotate_opts is not None:
                _print_rank0(
                    "[titan] FINAL token_loss "
                    + " ".join(
                        f"{k}={final_val[k]:.4f}"
                        for k in ("asst_loss", "errhead_auroc", "errhead_loss", "unlik_loss")
                        if k in final_val
                    )
                    + f" calls={int(final_val.get('errhead_calls', 0))}"
                )

    # Recurrent-depth arms: score the same held-out windows at each loop count.
    # Ranking stays on val_lm_loss above (default eval depth, FLOP-matched); the
    # sweep is what Parcae's L(T) = L_inf + Z*exp(-z*T/mu) fit and the oracle
    # per-token-exit analysis read. Rank-local means, like val_lm_loss itself.
    val_by_iters: Dict[str, Optional[float]] = {}
    val_iters: Optional[int] = None
    _rec = next(
        (m for m in unwrap_ddp(model).modules() if getattr(m, "use_recurrent_depth", False)),
        None,
    )
    if _rec is not None:
        _rcfg = _rec.config
        val_iters = sample_recursion_count(
            int(_rcfg.recurrent_min_iters),
            int(_rcfg.recurrent_max_iters),
            training=False,
            eval_iters=int(getattr(_rcfg, "recurrent_eval_iters", 0)),
        )
        sweep_max = int(config.eval.recurrent_iters_sweep_max)
        if sweep_max == 0:
            sweep_max = 2 * int(_rcfg.recurrent_max_iters)
        if config.eval.enabled and not is_sft and sweep_max > 0:
            for t in range(1, sweep_max + 1):
                _rec._recurrent_force_iters = t
                try:
                    val_by_iters[str(t)] = evaluate().get("lm_loss")
                finally:
                    _rec._recurrent_force_iters = None
            # Flat scalar keys outside "val/", and no "final" flag: anything that
            # takes the last final row as THE final eval must keep finding the
            # default-depth one.
            payload = {
                "step": config.training.max_steps,
                **{f"val_sweep/lm_loss_T{t}": v for t, v in val_by_iters.items()},
                "val_sweep/ranked_T": val_iters,
            }
            log_history.append(payload)
            if is_rank_zero():
                jsonl.log(payload)
            _print_rank0(
                f"[titan] recurrent depth sweep (val_lm_loss ranks at T={val_iters}): "
                + " ".join(
                    f"T={t}:{v:.4f}" if v is not None else f"T={t}:nan"
                    for t, v in val_by_iters.items()
                )
            )

    # Final checkpoint. Ensure TST flag is off before saving so the checkpoint
    # is a standard-NTP checkpoint regardless of where the loop exited.
    if tst_controller is not None:
        tst_controller.end(unwrap_ddp(model))
    mat_ffn_warmup.end(unwrap_ddp(model))
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        step=config.training.max_steps,
        config=config,
        log_history=log_history,
    )
    barrier()

    # One-line machine-readable verdict per run. Without this the sweep analysis
    # has to re-derive a final loss from noisy per-step training losses, which is
    # exactly what produced the bogus "more tokens = worse loss" reading earlier.
    if is_rank_zero():
        # Steady-state step time and the memory ceiling, so a shape/cost run does
        # not have to be read back out of the slurm log by eye. Step 1 is compile
        # (38 s at 64K, 57 s at 128K against a ~2.4 s steady state) and the long
        # shapes are still warming at step 10 -- dynamo recompiles per distinct
        # max_seqlen -- so a mean over all steps is meaningless. Median of the
        # LATTER HALF of the logged steps is the honest number; peak memory is a
        # max, because a ceiling is what decides whether a shape fits at all.
        _steps = [h for h in log_history if "step_time_ms" in h]
        _tail = _steps[len(_steps) // 2 :] or _steps
        _ms = sorted(h["step_time_ms"] for h in _tail)
        summary = {
            "run_name": config.job.run_name,
            "model_size": config.model.size,
            "seq_len": config.data.seq_len,
            "tokens_per_step": tokens_per_step,
            "step_time_ms": (_ms[len(_ms) // 2] if _ms else None),
            "step_time_ms_n": len(_ms),
            "peak_mem_gb": (
                max((h["peak_mem_gb"] for h in _steps if "peak_mem_gb" in h), default=None)
            ),
            "optimizer": config.optimizer.name,
            "lr": config.optimizer.lr,
            "lr_adam": config.optimizer.lr_adam,
            "max_steps": config.training.max_steps,
            "tokens": config.training.max_steps * tokens_per_step,
            "active_params": n_active,
            # Unique parameters. Differs from active_params on looped arms
            # (active bills the core at E[iters], so a FLOP-matched qwen3_like_rec
            # reports its dense parent's N) and on routed/nested arms. An L(N,D)
            # fit must say which one it used.
            "total_params": n_params,
            "val_loss": final_val.get("loss"),
            # Null on an SFT run, by design: the sweep ranks on val_lm_loss, and
            # a masked loss sits several nats below an unmasked FineWeb loss, so
            # an SFT number in this key would read as the best arm in the grid.
            "val_lm_loss": final_val.get("lm_loss"),
            "val_sft_loss": final_val.get("sft_loss"),
            "val_loss_tokens": final_val.get("loss_tokens"),
            # [token_loss] runs only (absent otherwise). val_lm_loss above stays
            # the UNWEIGHTED loss; val_asst_loss is the like-for-like number when
            # w_tool != 1 changes what the general loss is trained on.
            **(
                {
                    "token_loss": {
                        "w_tool": float(tl_cfg.w_tool),
                        "errhead_weight": float(tl_cfg.errhead_weight),
                        "unlik_weight": float(tl_cfg.unlik_weight),
                        "unlik_mode": str(tl_cfg.unlik_mode),
                    },
                    "val_asst_loss": final_val.get("asst_loss"),
                    "val_errhead_auroc": final_val.get("errhead_auroc"),
                    "val_errhead_calls": final_val.get("errhead_calls"),
                    "val_errhead_pos_frac": final_val.get("errhead_pos_frac"),
                    "val_unlik_loss": final_val.get("unlik_loss"),
                }
                if annotate_opts is not None
                else {}
            ),
            "objective": "sft" if is_sft else "lm",
            "val_sequences": int(final_val.get("sequences", 0)),
            # Recurrent-depth arms only (absent elsewhere): the loop count
            # val_lm_loss was scored at, and the same windows at T = 1..N.
            **(
                {"val_lm_loss_iters": val_iters, "val_lm_loss_by_iters": val_by_iters}
                if val_iters is not None
                else {}
            ),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        _print_rank0(f"[titan] wrote {output_dir / 'summary.json'}")

    if wandb_run is not None:
        wandb_run.finish()
    _print_rank0("[titan] training complete")
