# torchtitan_train

Single-node DDP pretraining loop for UMichSLM. Runs alongside the existing
LLaMA-Factory path (`scripts/llamafactory_train.py`); does not replace it.

## Quick start

```bash
# Single GPU
python scripts/torchtitan_train.py configs/torchtitan/umichslm_pretrain_baseline.toml

# 4-GPU DDP via torchrun
torchrun --standalone --nproc_per_node=4 \
    scripts/torchtitan_train.py configs/torchtitan/umichslm_pretrain_baseline.toml
```

## What's here

- `launch.py` — torchrun-aware entrypoint. Sets up the NCCL process group,
  parses TOML, hands off to `train.run`.
- `train.py` — the actual training loop. Optimizer, LR schedule, DDP wrap,
  dataloader, throughput tracking, HF-format checkpointing.

The model adapter (`UMichSLMTitanWrapper`), TOML schema, FineWeb dataloader,
checkpoint helpers, and throughput instrumentation live under
`src/umich_slm/titan/` so they ship with the package.

## TOML schema

See `src/umich_slm/titan/config_schema.py` for the dataclass definitions.
Required sections: `[job]`, `[model]`, `[data]`, `[training]`. Optional
(with defaults): `[optimizer]`, `[lr_scheduler]`, `[parallelism]`,
`[logging]`, `[checkpoint]`.

Field-by-field mapping from the existing LLaMA-Factory YAMLs:

| LF YAML | Titan TOML |
|---|---|
| `model_name_or_path` | `model.init_from` |
| `cutoff_len` | `data.seq_len` |
| `dataset` + `dataset_dir` | `data.dataset_dir` (+ `train_arrow` / `val_arrow`) |
| `per_device_train_batch_size` | `training.per_device_batch_size` |
| `gradient_accumulation_steps` | `training.gradient_accumulation_steps` |
| `learning_rate` | `optimizer.lr` |
| `lr_scheduler_type` + `lr_scheduler_kwargs` | `lr_scheduler.{name,warmup_steps,stable_steps,decay_steps}` |
| `bf16` | `training.bf16` |
| `gradient_checkpointing` | `training.gradient_checkpointing` |
| `max_steps` | `training.max_steps` |
| `seed` | `job.seed` |

## Output layout

```
outputs/titan/<run_name>/
├─ metrics.jsonl                     # one JSON line per logged step
├─ checkpoint-500/                   # HF safetensors + sidecars
│   ├─ model.safetensors
│   ├─ config.json
│   ├─ tokenizer files (if bundled)
│   ├─ optimizer.pt                  # opt + scheduler + step + RNG
│   ├─ titan_config.json             # resolved TOML for this run
│   └─ trainer_state.json            # mirrors HF Trainer's format
└─ checkpoint-1000/ ...
```

Checkpoints are HF-compatible: load with `AutoModelForCausalLM.from_pretrained(...)`
and run the existing eval scripts (`compute_val_loss.py`, `run_eval.py`, etc.)
without modification.
