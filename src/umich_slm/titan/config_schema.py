"""TOML config schema for the torchtitan training path.

Sections mirror what LLaMA Factory exposes through YAML, but the surface is
trimmed to the fields that are actually used by the titan trainer (no SFT/DPO,
no FSDP2/TP/PP). Unknown TOML fields raise ``ValueError`` so typos don't
silently fall through.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


try:  # py>=3.11
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - py<3.11
    import tomli as _toml  # type: ignore[no-redef]


@dataclass
class JobConfig:
    output_dir: str
    run_name: str
    seed: int = 42


@dataclass
class ModelConfig:
    size: str
    init_from: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    dataset_dir: str
    train_arrow: str = "dataset/train.arrow"
    val_arrow: str = "dataset/validation.arrow"
    seq_len: int = 2048
    text_column: str = "text"
    num_workers: int = 8
    tokenizer_path: Optional[str] = None
    shuffle_buffer: int = 10_000


@dataclass
class OptimizerConfig:
    name: str = "adamw"  # adamw | adamw_fused
    lr: float = 3.0e-4
    weight_decay: float = 0.1
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1.0e-8


@dataclass
class LRSchedulerConfig:
    name: str = "warmup_stable_decay"  # cosine | warmup_stable_decay
    warmup_steps: int = 191
    stable_steps: int = 3242
    decay_steps: int = 381
    min_lr_ratio: float = 0.0
    decay_type: str = "cosine"  # cosine | linear


@dataclass
class TrainingConfig:
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = 3814
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    torch_compile: bool = False
    float8: bool = False
    deltanet_compile_safe: bool = True


@dataclass
class ParallelismConfig:
    dp_replicate_degree: int = 1


@dataclass
class LoggingConfig:
    log_every: int = 10
    report_to: str = "wandb"  # wandb | none
    wandb_project: str = "umich-slm-titan"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    metrics_jsonl: str = "metrics.jsonl"


@dataclass
class CheckpointConfig:
    save_every: int = 500
    keep_last: int = 5
    format: str = "hf"  # hf | dcp (dcp reserved for future FSDP2 work)
    eval_every: int = 1000
    resume_from: Optional[str] = None


@dataclass
class StagingConfig:
    """Layer-stacked pretraining configuration.

    When ``mode == "in_process"`` and ``stages`` lists 2+ stage-spec JSON files,
    the torchtitan trainer runs all stages in one launch with in-memory
    transitions between them (no checkpoint round-trip). The first stage uses
    the model definition from ``[model]``; subsequent stages are built from
    the previous stage's live model + optimizer via
    ``umich_slm.staging.build_stage_n_model`` and
    ``umich_slm.staging.migrate_optimizer_state``.

    When ``mode == "external"`` (or this section is omitted), the trainer runs
    a single stage with no transitions — equivalent to existing behavior. In
    that mode, multi-stage runs are orchestrated externally by the LLaMA-Factory
    sbatch chain that calls ``python -m umich_slm.staging.transition`` between
    stage runs.

    ``max_steps_per_stage`` lets each stage have a different step budget; if a
    list of N values is provided, it overrides ``training.max_steps`` per stage.
    The 5/25/70 stage allocation from the plan would set this to e.g.
    ``[100, 500, 1400]`` (numbers vary by pilot).
    """

    mode: str = "external"  # "external" | "in_process"
    stages: List[str] = field(default_factory=list)  # paths to StageSpec JSONs
    init_strategy: str = "exact_copy"  # "exact_copy" | "llama_pro_identity"
    run_probes_between_stages: bool = False
    max_steps_per_stage: List[int] = field(default_factory=list)
    lr_per_stage: List[float] = field(default_factory=list)


@dataclass
class TitanConfig:
    job: JobConfig
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    staging: StagingConfig = field(default_factory=StagingConfig)


_SECTION_TYPES = {
    "job": JobConfig,
    "model": ModelConfig,
    "data": DataConfig,
    "training": TrainingConfig,
    "optimizer": OptimizerConfig,
    "lr_scheduler": LRSchedulerConfig,
    "parallelism": ParallelismConfig,
    "logging": LoggingConfig,
    "checkpoint": CheckpointConfig,
    "staging": StagingConfig,
}


def _build_section(cls, raw: Dict[str, Any]):
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown TOML fields in [{cls.__name__}]: {sorted(unknown)}. "
            f"Allowed: {sorted(known)}"
        )
    return cls(**raw)


def load_toml_config(path: str | Path) -> TitanConfig:
    """Parse a TOML file into a validated ``TitanConfig``.

    Required sections: ``[job]``, ``[model]``, ``[data]``, ``[training]``.
    Other sections fall back to dataclass defaults if omitted.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        raw = _toml.load(fh)

    unknown_sections = set(raw) - set(_SECTION_TYPES)
    if unknown_sections:
        raise ValueError(
            f"Unknown TOML sections: {sorted(unknown_sections)}. "
            f"Allowed: {sorted(_SECTION_TYPES)}"
        )

    for required in ("job", "model", "data", "training"):
        if required not in raw:
            raise ValueError(f"Missing required TOML section: [{required}]")

    sections: Dict[str, Any] = {}
    for name, cls in _SECTION_TYPES.items():
        if name in raw:
            sections[name] = _build_section(cls, raw[name])

    return TitanConfig(**sections)


# ---------------------------------------------------------- CLI overrides
def _coerce_value(field_type, raw: str) -> Any:
    """Best-effort string→typed coercion driven by the dataclass field type."""
    import typing

    # Unwrap Optional / Union[T, None]: pick the first non-None arg.
    origin = typing.get_origin(field_type)
    if origin is typing.Union:
        args = [a for a in typing.get_args(field_type) if a is not type(None)]
        if args:
            field_type = args[0]
            origin = typing.get_origin(field_type)

    if field_type is bool:
        low = raw.strip().lower()
        if low in {"true", "1", "yes", "y", "on"}:
            return True
        if low in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot parse {raw!r} as bool")
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    if field_type is str:
        return raw
    if origin is list:
        # Comma-separated list, item type taken from List[T] arg (default str).
        item_args = typing.get_args(field_type) or (str,)
        item_t = item_args[0]
        return [_coerce_value(item_t, v.strip()) for v in raw.split(",") if v.strip()]
    if origin is dict:
        # Lightweight ``k=v,k=v`` parsing for nested dicts (e.g. model.overrides).
        out: Dict[str, Any] = {}
        for entry in raw.split(","):
            if not entry.strip():
                continue
            k, _, v = entry.partition("=")
            out[k.strip()] = v.strip()
        return out
    # Fallback: use the raw string and let the dataclass surface a TypeError.
    return raw


def apply_overrides(config: TitanConfig, overrides: Iterable[str]) -> TitanConfig:
    """Apply ``section.field=value`` overrides to a parsed ``TitanConfig``.

    Used by the ``torchrun`` launcher so a single TOML can drive multiple
    GPU-count or variant-specific runs without forking the file. Mutates the
    config in place and returns it.
    """
    import typing as _t

    for entry in overrides:
        if "=" not in entry:
            raise ValueError(
                f"Override {entry!r} must be in the form 'section.field=value'"
            )
        key, _, value = entry.partition("=")
        if "." not in key:
            raise ValueError(
                f"Override key {key!r} must include a section prefix "
                f"(e.g. 'training.gradient_accumulation_steps')"
            )
        section_name, _, field_name = key.partition(".")
        section = getattr(config, section_name, None)
        if section is None or section_name not in _SECTION_TYPES:
            raise ValueError(
                f"Unknown section {section_name!r}; allowed: {sorted(_SECTION_TYPES)}"
            )
        # ``from __future__ import annotations`` turns field types into strings;
        # ``get_type_hints`` resolves them back to real type objects.
        hints = _t.get_type_hints(type(section))
        if field_name not in hints:
            raise ValueError(
                f"Unknown field {field_name!r} on [{section_name}]; "
                f"allowed: {sorted(hints)}"
            )
        coerced = _coerce_value(hints[field_name], value)
        setattr(section, field_name, coerced)
    return config
