"""TOML config schema for the torchtitan training path.

Sections mirror what LLaMA Factory exposes through YAML, but the surface is
trimmed to the fields that are actually used by the titan trainer (no DPO,
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
    # HF attention backend: "flash_attention_2" | "sdpa" | "eager". None lets the
    # config/HF default decide (sdpa). Set via `--set model.attn_implementation=...`;
    # flash_attention_2 silently falls back to sdpa if flash-attn isn't importable.
    attn_implementation: Optional[str] = None


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
    # "lm" reads the Arrow text column and trains on every token (pretraining).
    # "sft" reads Harmony jsonl.gz shards through titan.sft_data and trains only
    # on the assistant spans, which requires an [sft] section. The two differ in
    # what the loss DENOMINATOR is, so they are not interchangeable: an sft run
    # written into the lm metric keys would rank against pretraining arms on a
    # masked loss that sits several nats lower.
    kind: str = "lm"
    # Replay: a second Arrow pool mixed into the TRAIN stream (never validation).
    # Each packed window comes whole from one pool, interleaved so that exactly
    # ``replay_frac`` of windows (hence tokens, since all are seq_len) are replay.
    # Built for the 32K extension stage, whose long-only pool cost +0.91 nats at 4K
    # with no replay (2026-10-22). lm only; 0 disables.
    replay_dataset_dir: Optional[str] = None
    replay_arrow: str = "dataset/train.arrow"
    replay_frac: float = 0.0


@dataclass
class SFTConfig:
    """Masked-loss SFT. Required when ``[data].kind = "sft"``, forbidden otherwise.

    ``train_shards``/``val_shards`` are glob patterns over the prepared Harmony
    shards (``scripts/sft_prep/build_mix.py`` output), not Arrow directories.
    """

    train_shards: List[str] = field(default_factory=list)
    val_shards: List[str] = field(default_factory=list)
    # "skip" drops a conversation longer than seq_len; "truncate_middle" keeps
    # head+tail. skip is the default because truncating a trajectory discards the
    # final answer, which is usually the part worth training on -- and it is what
    # makes the progressive 4K->16K->32K chain a length curriculum rather than
    # three passes over the same data.
    long_policy: str = "skip"
    # {"source_name": n} yields that source n times per pass.
    source_epochs: Dict[str, int] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    name: str = "adamw"  # adamw | adamw_fused | adamh | muonh | muonw
    lr: float = 3.0e-4
    # AdamH and MuonW. The Delphi recipe runs TWO learning rates: η_proj (this is
    # ``lr``) on the Hyperball-constrained projection matrices, and a much
    # smaller η_adam on embeddings / norms / biases. Leaving this None applies
    # ``lr`` to both groups, which trains the embedding/norm/bias group ~7.5x
    # too hot at the sweep's calibration. Under muonw, ``lr`` drives the Muon
    # matrices and this drives the AdamW remainder (embeddings, head, norms,
    # banks, routers); None = ``lr``.
    lr_adam: Optional[float] = None
    weight_decay: float = 0.1
    # Scale the nested FFN's weight decay by how often each row is actually
    # trained (100/40/18/6% across tiers under the default prior). Decoupled
    # decay otherwise shrinks every row every step regardless of use, and the
    # WIDEST block -- the largest, 1536 of 3072 columns -- gets 16.7x fewer
    # gradient updates to push back with. Under the measured collapse it was
    # 0.02%, i.e. essentially nothing resisting.
    mat_ffn_freq_decay: bool = False
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1.0e-8
    # LR multiplier for the swa_registers bank + read gates ("register_tokens",
    # "register_gate" params) — an LR-calibration instrument for the registers
    # arch, NOT part of the Delphi recipe. AdamW path only; adamh/muonh ignore
    # it (their grouping is the recipe's). 1.0 = no separate group.
    register_lr_mult: float = 1.0
    # MuonW under DDP: shard Newton-Schulz across ranks and all-gather the bf16
    # update (replicas stay bitwise identical). False = every rank runs NS on
    # every matrix with no communication; bench A/B only.
    muon_ns_shard: bool = True


@dataclass
class LRSchedulerConfig:
    name: str = "warmup_stable_decay"  # cosine | warmup_stable_decay
    warmup_steps: int = 191
    stable_steps: int = 3242
    decay_steps: int = 381
    min_lr_ratio: float = 0.0
    decay_type: str = "cosine"  # cosine | linear


@dataclass
class TSTConfig:
    """Token-Superposition Training (Peng et al., arXiv:2605.06546).

    When ``enabled``, phase (i) groups contiguous tokens into bags of size
    ``bag_size`` (mean-pooled embeddings + multi-hot CE on the next bag); at
    step ``int(max_steps * recovery_ratio)`` the controller flips
    ``model.config.tst_active`` off and the loop slices each batch back to
    ``data.seq_len`` for the rest of the stage (phase ii — standard NTP).

    To preserve the paper's equal-FLOPs-per-step comparison, the data loader
    runs at ``data.seq_len * bag_size`` for the whole stage when
    ``extend_seq_len_for_equal_flops`` is True; phase (ii) just truncates.
    """

    enabled: bool = False
    bag_size: int = 6
    recovery_ratio: float = 0.3
    loss_weighting: str = "uniform"  # "uniform" | "power_law"
    extend_seq_len_for_equal_flops: bool = True


@dataclass
class TrainingConfig:
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = 3814
    max_grad_norm: float = 1.0
    # Embedding-healing phase for warm-started runs. For the first N steps only
    # `embed_tokens` and `lm_head` receive gradients; the body is frozen. This
    # exists because the dominant cost of a donor warm start is the *vocabulary*,
    # not the body: swapping a SuperBPE-composed embedding table into an
    # otherwise intact Qwen3-0.6B moves it from 0.7381 to 2.9114 bpb (random
    # embeddings: 4.3726). FOCUS/WECHSEL/ZeTT all close most of that gap with a
    # short embedding-only phase -- ZeTT reports under 1B tokens. 0 disables it.
    # Single-rank only: frozen params produce no gradient, which DDP rejects
    # without find_unused_parameters.
    freeze_body_steps: int = 0
    # Soft alternative to freeze_body_steps: the borrowed (donor) body trains from
    # step 0 at borrowed_lr_mult x the scheduled LR, held for borrowed_lr_hold_steps,
    # then ramped log-linearly to 1x over borrowed_lr_ramp_steps. `embed_tokens` and
    # `lm_head` (the re-composed tables) always get the plain schedule. Healing's
    # hard unfreeze at peak LR was a cliff (olmoheal train loss 5.59 -> 16.39 at
    # step 1,525, a 90x grad-norm shock against an embedding-only EMA, and the
    # body's Adam moments empty); here the body's moments and the guard's EMA
    # both track the full model throughout. 1.0 disables it. AdamW only.
    borrowed_lr_mult: float = 1.0
    borrowed_lr_hold_steps: int = 0
    borrowed_lr_ramp_steps: int = 0
    bf16: bool = True
    gradient_checkpointing: bool = True
    torch_compile: bool = False
    float8: bool = False
    # torchao recipe for float8 training. None keeps torchao's default
    # (tensorwise dynamic scaling). "rowwise" holds a scale per row --
    # slightly slower, and it recovers most of the accuracy tensorwise
    # loses on outlier-heavy activations, which is why production FP8
    # pretraining has largely moved to it. Unmeasured on our shapes until
    # the RTX PRO 6000 job reports; no local card can run FP8 at all.
    float8_recipe: Optional[str] = None
    # Leave DeltaNet decoder layers eager. Default OFF since 2026-10-20: FLA >= 0.5
    # fences its own kernels with torch.compiler.disable, so compiling the layer
    # only fuses the norms/FFN/projections around them, and skipping it left 75%
    # of a 3:1 hybrid uncompiled. Set true to get the old behaviour back.
    deltanet_compile_safe: bool = False
    # DDP's find_unused_parameters. True (the historical default) walks the autograd
    # graph after every synced forward to find parameters that will get no gradient;
    # MoD/MTP/routed-tier arms need it. The 2026-10-26 prod-scale bench found NO unused
    # parameters in qwen3_like / deltanet_mla* / qwen3_5_like_prope, so those can set it
    # false. A wrong false fails loudly (DDP raises on the next step), never silently.
    ddp_find_unused_parameters: bool = True
    # DDP static_graph: the autograd graph is identical every step, so DDP records
    # the gradient-ready order on step 1 and buckets/launches all-reduces in that
    # order (and needs no unused-parameter walk). Only for arms with no data-
    # dependent routing (MoD/MoE/mat_ffn tiers/recursion are NOT static).
    ddp_static_graph: bool = False
    # Gradient all-reduce compression: "none" (fp32 grads on the wire) or "bf16"
    # (torch's bf16_compress_hook: cast, reduce, cast back -- half the PCIe bytes,
    # slightly lossy, so A/B the loss before adopting it).
    ddp_comm_hook: str = "none"
    # Dynamo's DDPOptimizer splits each compiled block at gradient-bucket boundaries
    # so all-reduces overlap backward, but ONLY while DDP.forward is on the stack. A
    # grad-checkpoint recompute runs in backward, outside it; if anything forces a
    # recompile there it compiles the UNSPLIT block and the saved tensors no longer
    # line up (CheckpointError, every TITAN_DDP_SINGLE_RANK cell of GL 64806691,
    # dense included). False sets torch._dynamo.config.optimize_ddp=False for every
    # DDP run with compile + grad-ckpt: no split, so nothing to disagree, at the cost
    # of comm/compute overlap. Window-LSE arms already force it off (train.py).
    ddp_compile_split: bool = True
    # ROOT-CAUSE fix for the scribe/cmod A6000 compile crashes: disable Inductor's
    # POINTWISE autotuner. The MLA flash path's V-pad-then-slice
    # (attn_output[..., :128] off a 192-padded flash output) codegens an unmasked
    # stride-gather `_to_copy_slice_view`; the pointwise autotuner benchmarks a
    # config that over-reads the buffer -> illegal memory access (scribe) / NaN
    # (cmod). Throughput-neutral (pointwise kernels are memory-bound; matmul
    # autotune is untouched). See wrapper.apply_compile.
    safe_pointwise_autotune: bool = True
    # Fallback escape hatch (default OFF now that safe_pointwise_autotune fixes the
    # root cause). Leaves window-LSE (sink/scribe) layers eager for the MoD
    # gather-runner arms -- dodges the same fault but costs ~17% throughput on
    # those arms. Mirrors deltanet_compile_safe.
    window_lse_compile_safe: bool = False
    # Per-block compile of the dynamic-compute modules running OUTSIDE the
    # decoder layers (MoP lane blocks, batched trunk variants) — the mopd
    # eager-dispatch fix (2026-08-18). Escape hatch mirrors the knobs above;
    # suppress_errors already degrades any backend failure to eager.
    compile_dyncompute_blocks: bool = True
    # Grad-spike guard: skip the optimizer step when the pre-clip grad norm is
    # non-finite or exceeds grad_spike_factor x an EMA of recent healthy norms.
    # Absorbs the rare tail-event blow-ups that beta2=0.9999's ~10k-step variance
    # horizon otherwise turns permanent (esp. the spiky-gradient bank arms). The
    # EMA warms for grad_spike_warmup steps before the ratio test arms.
    grad_spike_skip: bool = True
    grad_spike_factor: float = 20.0
    grad_spike_warmup: int = 100
    # A SUSTAINED skip streak is not a spike -- it is the guard refusing to
    # follow a permanent shift in gradient SCALE, and as built that wedges the
    # run forever at a flat loss while still looking healthy. Three arms were
    # lost this way (swa_deltabank 10m/50m, think_suffix ar10m); think_suffix
    # still wrote a summary.json whose val_lm_loss read as "bad arch" rather
    # than "frozen at step 14,240 of 27,432". After this many CONSECUTIVE skips
    # the EMA is re-seeded to the current norm and training resumes. Safe
    # because max_grad_norm clips every applied update anyway, so adopting the
    # new scale cannot produce a large step -- skipping was the strictly more
    # conservative choice, and it is the one that lost the runs.
    grad_spike_recover_after: int = 50
    # If re-seeding does not settle things, the run is genuinely sick. Abort
    # loudly rather than emit a frozen summary that looks like a real result.
    grad_spike_max_rewarms: int = 5
    # Drain the mat_ffn on-device tier counters every N steps and log realized
    # occupancy. 0 disables. Cheap: one small D2H per N steps, no per-step sync.
    mat_ffn_stats_interval: int = 500
    # State/prediction separation phasing (modules/pause_stream.py). Fraction of
    # the schedule to train WITHOUT the prediction stream before switching it
    # on. 0 = separated from step 0 (the paper's "full pause", 1.57x). The
    # paper's cheap variants are 0.425 and 0.75, which recover 94% and 71% of
    # the gain at 1.33x and 1.14x. e_pause is the only tensor that initialises
    # fresh at the switch, so every other piece of training state carries over.
    # Inert unless the model has a pause stream.
    pause_phase_start_frac: float = 0.0
    # mtp_phase_start_frac (2026-10-31): the same phasing for multi-token prediction.
    # The MTP loss (and its forward) is off for the first fraction of max_steps and on
    # for the rest, so MTP acts as a late-training objective. 0 = MTP from step 0.
    # Inert unless the model has MTP heads (use_mtp).
    mtp_phase_start_frac: float = 0.0
    tst: TSTConfig = field(default_factory=TSTConfig)


@dataclass
class ParallelismConfig:
    dp_replicate_degree: int = 1
    # "ddp" (full replicas, the default) or "fsdp" (FSDP2 fully_shard per decoder
    # block + the root: params, grads and optimizer state sharded across ranks).
    # FSDP is what lets a ~1B model train at 32K on 46 GB cards. AdamW only, and it
    # cannot resume or init-load yet (it refuses rather than silently restarting);
    # its checkpoints are ordinary full HF checkpoints that a DDP run can resume.
    data_parallel: str = "ddp"
    # FSDP compute dtype: "fp32" keeps fp32 params under bf16 autocast, i.e. the
    # same numerics as DDP; "bf16" casts params to bf16 for compute
    # (MixedPrecisionPolicy, fp32 reduce), halving all-gathers and the residual
    # stream's checkpointed activations.
    fsdp_param_dtype: str = "fp32"
    # ZeRO-3 (True): blocks free their gathered params after forward and gather
    # again in backward. False keeps them gathered (ZeRO-2-like: less comm, more memory).
    fsdp_reshard_after_forward: bool = True


@dataclass
class LoggingConfig:
    log_every: int = 10
    report_to: str = "wandb"  # wandb | none
    wandb_project: str = "umich-slm-titan"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    metrics_jsonl: str = "metrics.jsonl"
    # Log optim_ms: the optimizer step's GPU time (one extra CUDA sync per step).
    time_optimizer: bool = False


@dataclass
class CheckpointConfig:
    save_every: int = 500
    keep_last: int = 5
    format: str = "hf"  # hf | dcp (dcp reserved for future FSDP2 work)
    eval_every: int = 1000
    resume_from: Optional[str] = None
    # Start a NEW run from another run's WEIGHTS (a checkpoint-{step} dir): tensors
    # only, fresh optimizer + LR schedule, step 0. For stages that change the
    # recipe under a trained model -- e.g. a 32K context extension of a 4096 grid
    # cell. Used ONLY when this run has no usable checkpoint of its own, so a
    # requeued job auto-resumes its own progress instead of re-initialising.
    # The model is still built from [model].size + overrides; the checkpoint's
    # config.json must agree with it except on the fields named below
    # (titan/checkpoint.py::init_config_mismatches), because a shape-neutral knob
    # would otherwise load cleanly into a different architecture.
    init_weights_from: Optional[str] = None
    init_weights_allow_config_diff: List[str] = field(default_factory=list)
    # Start a NEW run by GROWING a smaller same-family checkpoint into [model]
    # (umich_slm/warmstart/grow.py): "szp" = fresh init + grow_shrink * padded
    # donor, "fp" = function-preserving. Same resume rule as init_weights_from;
    # the donor config may differ only in the size fields (+ the allow list).
    grow_from: Optional[str] = None
    grow_operator: str = "szp"
    grow_shrink: float = 0.4


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
class EvalConfig:
    """Held-out validation on ``data.val_arrow``.

    The scaling sweep ranks architectures on held-out LM loss, so every knob here
    must be identical across runs for the numbers to be comparable: a fixed batch
    count, a fixed seed, and no shuffling — the same token windows, in the same
    order, for every architecture.
    """

    enabled: bool = True
    every: int = 0  # steps between evals; 0 = end of run only
    # Per-domain held-out bits-per-byte, logged next to val loss. Val loss says a
    # run is improving; BPB per bucket says where, which makes a data-mixture
    # change or a regression on code legible while the run is still going.
    # Empty path disables it, so existing sweeps are byte-identical.
    capability_eval_set: str = ""
    # Steps between capability evals; 0 means "whenever the val eval runs".
    # Kept separate from `every` because this costs more than a val pass and is
    # usually wanted less often.
    capability_every: int = 0
    # Fixed number of held-out SEQUENCES, not batches. The MTP architectures run at
    # per_device_batch_size=8 while the rest run at 32 (MTP's second logits tensor
    # would OOM otherwise), so a fixed batch count would silently hand them a
    # different — and 4x smaller — eval set. Window packing is deterministic given
    # (seed, seq_len) and independent of batch size, so fixing the sequence count
    # makes every architecture score the identical windows.
    sequences: int = 2048  # 2048 x 2048 tok = 4.2M held-out tokens
    seed: int = 1234  # fixed, and deliberately NOT job.seed (which varies per run)
    # Recurrent-depth arms only (no-op elsewhere): after the final eval, re-score
    # the same windows at every loop count 1..N and write summary.json's
    # val_lm_loss_by_iters. The ranking key val_lm_loss stays at the default
    # eval depth -- the FLOP-matched number -- since picking the best depth would
    # be selecting on the validation set at unequal cost. 0 -> N = 2 x
    # recurrent_max_iters (enough to see Parcae's saturation past the training
    # range); negative disables the sweep.
    recurrent_iters_sweep_max: int = 0


@dataclass
class TokenLossConfig:
    """Per-token loss signals read from Harmony structure at load time.

    Implemented by ``titan/harmony_annotate.py`` (labels) and
    ``modules/agentic_heads.py`` (losses). Every default is OFF, and with all of
    them off the loaders and the forward are the pre-existing code paths.
    Works for both ``[data].kind = "lm"`` and ``"sft"``.
    """

    # P6: loss weight on tool-result tokens (``<|start|>tool ... <|end|>``).
    # 1.0 = off. The LM objective becomes sum(w*CE)/sum(w); val_lm_loss stays
    # the UNWEIGHTED mean so it remains comparable across arms.
    w_tool: float = 1.0
    # Log the assistant-only CE (train and val, ``asst_loss`` / ``val_asst_loss``)
    # without changing the objective. Forced on by any knob below, since the
    # general val loss moves trivially when tool text is down-weighted.
    role_metrics: bool = False
    # P3b: BCE weight of the failed-call head at each ``<|call|>`` token. > 0
    # builds the head (model override ``use_errhead``) and logs val AUROC.
    errhead_weight: float = 0.0
    # P3c: unlikelihood weight on the failing call's tokens at the corrected
    # call's positions. ``unlik_mode``: "aligned" (every differing offset) or
    # "first_diverge" (only the first, where the two prefixes are identical).
    unlik_weight: float = 0.0
    unlik_mode: str = "aligned"

    @property
    def enabled(self) -> bool:
        return (
            bool(self.role_metrics)
            or float(self.w_tool) != 1.0
            or float(self.errhead_weight) > 0.0
            or float(self.unlik_weight) > 0.0
        )


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
    eval: EvalConfig = field(default_factory=EvalConfig)
    token_loss: TokenLossConfig = field(default_factory=TokenLossConfig)
    # None rather than a default instance: its ABSENCE is meaningful, because
    # validate_config uses it to cross-check [data].kind.
    sft: Optional[SFTConfig] = None


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
    "eval": EvalConfig,
    "token_loss": TokenLossConfig,
    "sft": SFTConfig,
}


def _build_section(cls, raw: Dict[str, Any]):
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    import typing as _t

    known_fields = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known_fields)
    if unknown:
        raise ValueError(
            f"Unknown TOML fields in [{cls.__name__}]: {sorted(unknown)}. "
            f"Allowed: {sorted(known_fields)}"
        )
    # Resolve string-form annotations (from `from __future__ import annotations`).
    hints = _t.get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for name, val in raw.items():
        f_type = hints.get(name, known_fields[name].type)
        if is_dataclass(f_type) and isinstance(val, dict):
            kwargs[name] = _build_section(f_type, val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


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

    return validate_config(TitanConfig(**sections))


# ---------------------------------------------------------- CLI overrides
def _coerce_scalar(raw: str) -> Any:
    """Coerce a value whose declared type is untyped (``Any``).

    ``model.overrides`` is ``Dict[str, Any]``, so the field type carries no
    per-value hint — but it feeds ``UMichSLMConfig``, whose fields *are* typed
    (``sliding_window: int``). A str there raises deep inside the model rather
    than at parse time, so infer the scalar type from the literal instead.
    """
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


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
        val_args = typing.get_args(field_type)
        val_t = val_args[1] if len(val_args) == 2 else typing.Any
        out: Dict[str, Any] = {}
        for entry in raw.split(","):
            if not entry.strip():
                continue
            k, _, v = entry.partition("=")
            v = v.strip()
            out[k.strip()] = _coerce_scalar(v) if val_t is typing.Any else _coerce_value(val_t, v)
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
            raise ValueError(f"Override {entry!r} must be in the form 'section.field=value'")
        key, _, value = entry.partition("=")
        if "." not in key:
            raise ValueError(
                f"Override key {key!r} must include a section prefix "
                f"(e.g. 'training.gradient_accumulation_steps')"
            )
        section_name, _, field_path = key.partition(".")
        section = getattr(config, section_name, None)
        if section is None or section_name not in _SECTION_TYPES:
            raise ValueError(f"Unknown section {section_name!r}; allowed: {sorted(_SECTION_TYPES)}")
        # Walk dotted ``a.b.c`` paths into nested dataclasses (e.g. training.tst.bag_size).
        path_parts = field_path.split(".")
        cursor = section
        for part in path_parts[:-1]:
            hints = _t.get_type_hints(type(cursor))
            if part not in hints:
                raise ValueError(
                    f"Unknown field {part!r} while resolving {key!r}; "
                    f"allowed at this level: {sorted(hints)}"
                )
            nxt = getattr(cursor, part, None)
            if nxt is None or not is_dataclass(type(nxt)):
                raise ValueError(
                    f"Cannot descend into {part!r} for override {key!r}: not a dataclass"
                )
            cursor = nxt
        leaf = path_parts[-1]
        hints = _t.get_type_hints(type(cursor))
        if leaf not in hints:
            raise ValueError(
                f"Unknown field {leaf!r} on {type(cursor).__name__}; " f"allowed: {sorted(hints)}"
            )
        coerced = _coerce_value(hints[leaf], value)
        # Dict-typed fields MERGE rather than replace, so repeated
        # ``--set model.overrides=k=v`` flags accumulate. Replacing would make
        # the last flag silently win, and packing several keys into one flag is
        # not an option: sbatch --export splits on commas.
        if isinstance(coerced, dict):
            existing = getattr(cursor, leaf, None)
            if isinstance(existing, dict):
                merged = dict(existing)
                merged.update(coerced)
                coerced = merged
        setattr(cursor, leaf, coerced)
    validate_config(config)
    return config


# ---------------------------------------------------------- cross-section checks
def validate_config(config: TitanConfig) -> TitanConfig:
    """Cross-section invariants that no single dataclass can express.

    Run at parse time AND after ``apply_overrides``: ``--set data.kind=sft`` on a
    TOML with no ``[sft]`` section parses cleanly and would otherwise reach the
    trainer, which would silently fall back to the Arrow loader and train an SFT
    run on unmasked pretraining text.
    """
    kind = str(config.data.kind).lower()
    if kind not in ("lm", "sft"):
        raise ValueError(f"[data].kind must be 'lm' or 'sft', got {config.data.kind!r}")

    frac = float(config.data.replay_frac)
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"[data].replay_frac must be in [0, 1), got {frac}")
    if frac > 0 and not config.data.replay_dataset_dir:
        raise ValueError("[data].replay_frac > 0 requires [data].replay_dataset_dir")
    if frac > 0 and kind == "sft":
        raise ValueError("[data].replay_frac is lm-only; an sft run has no Arrow replay path")

    tl = config.token_loss
    if float(tl.w_tool) < 0.0:
        raise ValueError(f"[token_loss].w_tool must be >= 0, got {tl.w_tool}")
    for name in ("errhead_weight", "unlik_weight"):
        if float(getattr(tl, name)) < 0.0:
            raise ValueError(f"[token_loss].{name} must be >= 0, got {getattr(tl, name)}")
    if tl.unlik_mode not in ("aligned", "first_diverge"):
        raise ValueError(
            f"[token_loss].unlik_mode must be 'aligned' or 'first_diverge', got {tl.unlik_mode!r}"
        )

    if kind == "sft":
        if config.sft is None:
            raise ValueError("[data].kind = 'sft' requires an [sft] section")
        if not config.sft.train_shards:
            raise ValueError("[sft].train_shards is empty; nothing to train on")
        if config.sft.long_policy not in ("skip", "truncate_middle"):
            raise ValueError(
                f"[sft].long_policy must be 'skip' or 'truncate_middle', "
                f"got {config.sft.long_policy!r}"
            )
        for name, n in (config.sft.source_epochs or {}).items():
            if int(n) < 1:
                raise ValueError(f"[sft].source_epochs[{name!r}] must be >= 1, got {n}")
    elif config.sft is not None:
        # An explicit error, not a quiet ignore: a leftover [sft] block under
        # kind="lm" means someone edited a TOML expecting SFT and got an
        # unmasked pretraining run whose metrics look entirely plausible.
        raise ValueError(
            "[sft] section present but [data].kind is 'lm'. Set data.kind = 'sft' "
            "or remove the [sft] section."
        )
    return config
