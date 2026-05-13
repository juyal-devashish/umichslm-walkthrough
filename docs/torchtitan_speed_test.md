# LLaMA-Factory vs torchtitan speed test

End-to-end workflow for the head-to-head pretraining throughput comparison.
Both frameworks train the same model on the same data with matched effective
batch / seq_len / precision; the throughput script then pulls per-step
tokens/sec out of each side's logs and prints a paired markdown table.

## Prerequisites

- Both training paths set up: see `torchtitan_train/README.md` for titan, the
  existing LLaMA-Factory configs in `configs/llama_factory/` for LF.
- The pretraining LF YAMLs have already been patched to emit
  `train_tokens_per_second` in `trainer_state.json` (search for
  `include_tokens_per_second: true` in `configs/llama_factory/umichslm_pretrain*.yaml`).
- A FineWeb-edu Arrow dataset prepared via `scripts/prepare_fineweb_dataset.py`,
  or any other dataset both frameworks can read.

## 1. Run the LLaMA-Factory variants

For each variant, sbatch the existing `local-slurm/small_<variant>_lr3e4_fineweb-1B.sh`.
Each writes its checkpoint and `trainer_state.json` under
`<output_dir>/`.

```bash
sbatch local-slurm/small_baseline_lr3e4_fineweb-1B.sh
sbatch local-slurm/small_mla_lr3e4_fineweb-1B.sh
sbatch local-slurm/small_moe_lr3e4_fineweb-1B.sh
sbatch local-slurm/small_mod_lr3e4_fineweb-1B.sh
sbatch local-slurm/small_mtp_lr3e4_fineweb-1B.sh
sbatch local-slurm/small_deltanet_gated_lr3e4_fineweb-1B.sh
sbatch local-slurm/small_deltanet_kda_lr3e4_fineweb-1B.sh
sbatch local-slurm/small_swa512_lr3e4_fineweb-1B.sh
sbatch local-slurm/small_zcrmsnorm_lr3e4_fineweb-1B.sh
sbatch local-slurm/small_pope_lr3e4_fineweb-1B.sh
```

## 2. Run the torchtitan variants

The torchtitan TOMLs in `configs/torchtitan/` mirror the LF batch sizing
(per-device + grad-accum) so the comparison is apples-to-apples.

```bash
sbatch local-slurm/titan_small_baseline_lr3e4_fineweb-1B.sh
sbatch local-slurm/titan_small_mla_lr3e4_fineweb-1B.sh
sbatch local-slurm/titan_small_moe_lr3e4_fineweb-1B.sh
sbatch local-slurm/titan_small_mod_lr3e4_fineweb-1B.sh
sbatch local-slurm/titan_small_mtp_lr3e4_fineweb-1B.sh
sbatch local-slurm/titan_small_deltanet_gated_lr3e4_fineweb-1B.sh
sbatch local-slurm/titan_small_deltanet_kda_lr3e4_fineweb-1B.sh
sbatch local-slurm/titan_small_swa512_lr3e4_fineweb-1B.sh
sbatch local-slurm/titan_small_zcrmsnorm_lr3e4_fineweb-1B.sh
sbatch local-slurm/titan_small_pope_lr3e4_fineweb-1B.sh
```

For multi-GPU scaling tests, also run the `_ddp{2,4,8}` variants.

## 3. Generate the comparison report

Once both sides have run (or even partially run — the script tolerates missing
runs and reports them as "no LF run" / "no titan run"):

```bash
python scripts/throughput_compare.py \
    --lf_runs /shared/4/projects/umich-slm/models \
    --titan_runs ./outputs/titan \
    --out docs/throughput_comparison_v1.md \
    --summary_plot docs/throughput_comparison_v1.png
```

This writes:

- `docs/throughput_comparison_v1.md` — markdown table with one row per variant
- `docs/throughput_comparison_v1.csv` — same data, machine-readable
- `docs/throughput_comparison_v1.png` — bar chart (skipped if matplotlib isn't
  installed)

### What's in the table

| column | meaning |
|---|---|
| variant | baseline / mla / moe / mod / mtp / deltanet_gated / deltanet_kda / swa512 / zcrmsnorm / pope |
| world | DDP world size (1 / 2 / 4 / 8) |
| eff_batch | per_device × grad_accum × world_size |
| seq_len | sequence length |
| LF tok/s | `train_tokens_per_second` from `trainer_state.json` |
| titan tok/s | tail-mean of `tokens_per_sec` in `metrics.jsonl` (skips first 20% as warmup) |
| speedup | titan tok/s ÷ LF tok/s; **bold** when ≥ 1.0× |
| LF step ms | `train_runtime / global_step × 1000` |
| titan step ms | tail-mean of `step_time_ms` |
| titan MFU % | tail-mean of `mfu_pct` (LF doesn't log MFU directly) |
| titan peak GB | max observed `peak_mem_gb` across the run |
| notes | warnings about effective_batch / seq_len / world_size mismatches |

## 4. Apples-to-apples invariants the script checks

- **effective_batch** must match (per_device × grad_accum × world_size). MTP
  uses 160 and PoPE uses 144 by design — both LF and titan use the same value
  for the same variant, so the *within-variant* comparison is still fair.
- **seq_len** must match (default 2048).
- **world_size** must match (run the same `_ddp{N}` variant on both sides).
- **bf16** must match.

Mismatches don't abort the report; they surface in the *notes* column so
reviewers can spot un-fair pairings.

## 5. Common gotchas

- **`LF tok/s = —` in the table**: the LF run finished before the YAML carried
  `include_tokens_per_second: true`. Re-run that variant on the patched YAML.
- **`titan tok/s = —`**: the run died before producing any `metrics.jsonl`
  rows (likely a CUDA OOM at step 0, or a config error). Check the slurm log.
- **Speedup < 1.0 on a known-fast variant**: confirm the runs actually used
  the same batch size — torchtitan with grad_accum overrides via `--set` but
  no matching LF override is the most common silent mismatch.
- **MoE/MoD show DDP overhead**: expected — those variants run with
  `find_unused_parameters=True`, which costs roughly 5-15% throughput. That
  cost is intrinsic to the architecture, not to torchtitan; it shows up
  symmetrically when LF runs the same variant under DDP.

## 6. Validation gates from the implementation plan

After Phase 4 of the plan, the speed-test report should answer:

1. **Same-step parity**: titan loss curve overlays LF baseline within ±1.5%
   (gated by Phase 1 work, not by this script).
2. **Headline numbers**: titan baseline ≥ LF baseline tokens/sec on the same
   GPU; if not, the breakdown column tells you whether that's a step-time
   regression vs a torch.compile-off regression.
3. **Variant coverage**: every P0 variant has a paired row with non-empty
   tokens/sec on both sides.
4. **Multi-GPU scaling** (Phase 2): the `_ddp{2,4,8}` runs of the baseline
   should show tokens/sec scaling ≥ 0.85× linear.

## 7. Re-running the comparison incrementally

The script is idempotent — re-running it picks up any new run subdirs in
either input root. Use it as a daily check while runs are in flight; it
ignores incomplete runs (no `trainer_state.json` summary entry / no
`metrics.jsonl` rows) and just lists them as missing.
