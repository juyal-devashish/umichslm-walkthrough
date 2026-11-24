"""Checkpointing in HuggingFace safetensors format.

Saves rank-0 only (DDP keeps full replicas; FSDP gathers to full first). The output
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
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from .config_schema import TitanConfig
from .parallelize import is_fsdp, is_rank_zero, unwrap_ddp


def _checkpoint_dir(output_dir: Path, step: int) -> Path:
    return output_dir / f"checkpoint-{step}"


def _serialize_config(config: TitanConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
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
        "token_loss",
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
    full_model_sd = full_optim_sd = None
    if is_fsdp(model):
        # Sharded params: every rank joins the all-gather, rank 0 writes. The
        # result is an ordinary full HF checkpoint; the optimizer goes out as a
        # full state dict keyed by parameter name (``optimizer_full_fqn``), which
        # load_checkpoint can hand to an unsharded run.
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            get_optimizer_state_dict,
        )

        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        full_model_sd = get_model_state_dict(model, options=opts)
        full_optim_sd = get_optimizer_state_dict(model, optimizer, options=opts)

    if not is_rank_zero():
        return None

    output_dir = Path(config.job.output_dir)
    ckpt_dir = _checkpoint_dir(output_dir, step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    inner = unwrap_ddp(model)
    # The wrapper's ``inner`` attribute is the HF ``UMichSLMForCausalLM``.
    hf_model = getattr(inner, "inner", inner)
    if full_model_sd is not None:
        prefix = "inner." if hf_model is not inner else ""
        hf_sd = {k[len(prefix):]: v for k, v in full_model_sd.items() if k.startswith(prefix)}
        hf_model.save_pretrained(ckpt_dir, state_dict=hf_sd, safe_serialization=True)
    else:
        hf_model.save_pretrained(ckpt_dir, safe_serialization=True)

    # Optimizer/scheduler/step/RNG sidecar.
    sidecar: Dict[str, Any] = {
        "step": step,
        "torch_rng_state": torch.get_rng_state(),
    }
    if full_optim_sd is not None:
        sidecar["optimizer_full_fqn"] = full_optim_sd
    else:
        sidecar["optimizer"] = optimizer.state_dict()
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


def _as_byte_tensor(state: Any) -> "torch.Tensor":
    """Coerce a round-tripped RNG state to a contiguous CPU uint8 ByteTensor."""
    if torch.is_tensor(state):
        return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(state, dtype=torch.uint8).contiguous()


def _restore_rng_state(sidecar: Dict[str, Any]) -> None:
    """Best-effort restore of torch (+CUDA) RNG state; warns instead of raising."""
    if "torch_rng_state" in sidecar:
        try:
            torch.set_rng_state(_as_byte_tensor(sidecar["torch_rng_state"]))
        except Exception as e:  # pragma: no cover - torch-version dependent
            print(f"[titan] warning: could not restore torch RNG state: {e}")
    if torch.cuda.is_available() and "cuda_rng_state_all" in sidecar:
        try:
            states = [_as_byte_tensor(s) for s in sidecar["cuda_rng_state_all"]]
            torch.cuda.set_rng_state_all(states)
        except Exception as e:  # pragma: no cover - torch-version dependent
            print(f"[titan] warning: could not restore CUDA RNG state: {e}")


def _load_model_weights(
    path: Path, model: torch.nn.Module, optional_prefixes: Tuple[str, ...] = ()
) -> None:
    """Copy a ``checkpoint-{step}`` directory's weights into ``model`` in place.

    Strict on genuinely missing or unexpected keys; tolerant of tied/aliased
    tensors that safetensors refused to serialize (see the comment below).
    Shape mismatches raise from ``load_state_dict`` itself.

    ``optional_prefixes`` (init path only) names auxiliary heads that may be
    absent on either side -- e.g. ``errhead.`` when a run adds the P3 failed-call
    head on top of a checkpoint trained without it. Those keys keep their fresh
    init (or are dropped) instead of aborting. Resume never passes any.
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
            # Weight-tying makes several state_dict names alias ONE tensor:
            # ``pathway_stage_share_lanes`` reuses stage 0's lane_stack /
            # shared_kv_proj across every later stage (modeling_umichslm.py), and
            # tied word embeddings alias lm_head to embed_tokens. safetensors
            # refuses to serialize aliased tensors, so only the first name is on
            # disk and the aliases surface here as "missing" — even though loading
            # the present name already populated them in-place. Treat a missing key
            # as harmless when its live tensor shares storage with a key that WAS
            # present; only raise on keys that are genuinely absent.
            live = hf_model.state_dict(keep_vars=True)
            present_ptrs = {
                live[k].data_ptr()
                for k in sd
                if k in live and live[k].device.type != "meta"
            }
            genuine = [
                k
                for k in missing
                if not (
                    k in live
                    and live[k].device.type != "meta"
                    and live[k].data_ptr() in present_ptrs
                )
                and not k.startswith(tuple(optional_prefixes))
            ]
            if genuine:
                raise RuntimeError(f"Missing keys when resuming: {genuine[:5]}...")
        unexpected = [k for k in unexpected if not k.startswith(tuple(optional_prefixes))]
        if unexpected:
            raise RuntimeError(f"Unexpected keys when resuming: {unexpected[:5]}...")
    else:
        # Use HF loader for shard layouts.
        loaded = type(hf_model).from_pretrained(path)
        hf_model.load_state_dict(loaded.state_dict())


# HF bookkeeping that legitimately differs between a checkpoint's config.json and a
# freshly built config (library version, save-time dtype, HF defaults emitted only
# on one side). Never architecture.
_CONFIG_BOOKKEEPING_KEYS = frozenset(
    {
        "_name_or_path",
        "architectures",
        "attn_implementation",
        "_attn_implementation",
        "_attn_implementation_autoset",
        "auto_map",
        "dtype",
        "torch_dtype",
        "transformers_version",
    }
)


_CONFIG_KEY_MIRRORS = {"rope_scaling": ("rope_parameters",)}
# Sub-keys added to a dict-valued field after older checkpoints were written,
# with the value those checkpoints actually ran at. 0f25858 put
# partial_rotary_factor into rope_parameters; every earlier run rotated all dims.
_CONFIG_NESTED_DEFAULTS = {"rope_parameters": {"partial_rotary_factor": 1.0}}
# Allowing a top-level key that transformers ALSO mirrors as one sub-key of a dict
# field exempts that sub-key only, never the whole dict. Named for the RoPE-base
# extension (2026-10-27, LONG_CONTEXT_PLAN L2: raise rope_theta instead of YaRN):
# allowing "rope_theta" must not also wave through a changed partial_rotary_factor
# or rope_type that rides in the same rope_parameters dict.
_CONFIG_SUBKEY_MIRRORS = {"rope_theta": ("rope_parameters", "rope_theta")}


def init_config_mismatches(
    ckpt_config: Dict[str, Any],
    live_config: Dict[str, Any],
    allowed: Optional[Iterable[str]] = None,
) -> Dict[str, Tuple[Any, Any]]:
    """Config fields where an init checkpoint and the live model DISAGREE.

    Compared only on keys present on BOTH sides: a key absent from an older
    config.json is a field added since, which takes its default. Returns
    ``{key: (checkpoint_value, live_value)}`` for every difference not in
    ``allowed`` or the HF bookkeeping set.

    WHY. ``init_weights_from`` builds the model from ``[model].size`` +
    overrides and then copies tensors in. A strict ``load_state_dict`` proves
    the SHAPES match, but a shape-neutral knob -- a window size, a router mode,
    ``swa_fuse_register_proj``, a preset edited since the checkpoint was
    trained -- would load cleanly into a different architecture and train on
    silently. Every intended change (a new ``max_position_embeddings`` and
    ``rope_scaling`` for a context extension, say) must therefore be named.
    """
    skip = _CONFIG_BOOKKEEPING_KEYS | set(allowed or ())
    # transformers>=5 mirrors rope_scaling into rope_parameters (plus rope_theta /
    # rope_type) on every config, so allowing one must allow its mirror.
    for key, mirrors in _CONFIG_KEY_MIRRORS.items():
        if key in skip:
            skip |= set(mirrors)
    drop_sub: Dict[str, set] = {}
    for key, (field, sub) in _CONFIG_SUBKEY_MIRRORS.items():
        if key in skip:
            drop_sub.setdefault(field, set()).add(sub)
    out: Dict[str, Tuple[Any, Any]] = {}
    for key in sorted(set(ckpt_config) & set(live_config)):
        if key in skip:
            continue
        a, b = ckpt_config[key], live_config[key]
        if key in _CONFIG_NESTED_DEFAULTS and isinstance(a, dict) and isinstance(b, dict):
            a = {**_CONFIG_NESTED_DEFAULTS[key], **a}
            b = {**_CONFIG_NESTED_DEFAULTS[key], **b}
        if key in drop_sub and isinstance(a, dict) and isinstance(b, dict):
            a = {k: v for k, v in a.items() if k not in drop_sub[key]}
            b = {k: v for k, v in b.items() if k not in drop_sub[key]}
        if a != b:
            # JSON turns tuples into lists; compare on the JSON form.
            if json.dumps(a, sort_keys=True, default=str) == json.dumps(
                b, sort_keys=True, default=str
            ):
                continue
            out[key] = (a, b)
    return out


def load_init_weights(
    path: Path,
    model: torch.nn.Module,
    *,
    allow_config_diff: Optional[Iterable[str]] = None,
    optional_prefixes: Tuple[str, ...] = (),
) -> Dict[str, Tuple[Any, Any]]:
    """Initialise a NEW run from another run's weights: tensors only.

    No optimizer state, no scheduler state, no RNG, no step -- the caller starts
    at step 0 with a fresh optimizer. This is what a context-extension stage
    wants (and all a weights-only copy of a finished cell can offer, since its
    ``optimizer.pt`` was not kept). Distinct from ``[model].init_from``, which
    goes through ``from_pretrained``: that path silently random-initialises any
    key it cannot find (tied aliases included) and takes its architecture from
    the checkpoint rather than from ``[model]``.

    Raises before touching the weights if the checkpoint's config.json disagrees
    with the live model on anything outside ``allow_config_diff`` (see
    :func:`init_config_mismatches`). Returns the ALLOWED differences, so the
    caller can log exactly what the new run changes.
    """
    path = Path(path)
    if not (path / "model.safetensors").exists():
        raise FileNotFoundError(f"init_weights_from: no model.safetensors in {path}")
    inner = unwrap_ddp(model)
    hf_model = getattr(inner, "inner", inner)
    cfg_path = path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"init_weights_from: no config.json in {path}")
    ckpt_cfg = json.loads(cfg_path.read_text())
    live_cfg = hf_model.config.to_dict()
    bad = init_config_mismatches(ckpt_cfg, live_cfg, allowed=allow_config_diff)
    if bad:
        lines = "\n".join(f"  {k}: checkpoint={a!r} live={b!r}" for k, (a, b) in bad.items())
        raise ValueError(
            f"init_weights_from={path}: live model config differs from the checkpoint on "
            f"fields not listed in checkpoint.init_weights_allow_config_diff:\n{lines}"
        )
    _load_model_weights(path, model, optional_prefixes=optional_prefixes)
    return init_config_mismatches(ckpt_cfg, live_cfg, allowed=None)


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
    _load_model_weights(path, model)

    sidecar_path = path / "optimizer.pt"
    if sidecar_path.exists() and (optimizer is not None or scheduler is not None):
        sidecar = torch.load(sidecar_path, map_location=map_location or "cpu", weights_only=False)
        if optimizer is not None and "optimizer" in sidecar:
            optimizer.load_state_dict(sidecar["optimizer"])
        elif optimizer is not None and "optimizer_full_fqn" in sidecar:
            # Written by an FSDP run: a full state dict keyed by parameter name.
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                set_optimizer_state_dict,
            )

            set_optimizer_state_dict(
                unwrap_ddp(model),
                optimizer,
                sidecar["optimizer_full_fqn"],
                options=StateDictOptions(full_state_dict=True),
            )
        if scheduler is not None and "scheduler" in sidecar:
            scheduler.load_state_dict(sidecar["scheduler"])
        # RNG-state restore is best-effort: it only affects exact reproducibility
        # of the post-resume data/dropout stream, not correctness. Some torch
        # versions return the round-tripped state as a plain tensor that
        # ``set_rng_state`` rejects ("RNG state must be a torch.ByteTensor"), so
        # coerce to a contiguous CPU uint8 ByteTensor and never let it abort resume.
        _restore_rng_state(sidecar)
        return int(sidecar.get("step", 0))

    return 0
