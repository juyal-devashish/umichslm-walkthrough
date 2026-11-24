# Evaluation Guide

This doc explains the offline evaluation scripts under `scripts/` and when
each one should be used. All scripts are additive — none of them modify the
training pipeline (`pretrain_fineweb.py` / LLaMA Factory) or the existing
`run_eval.py` downstream-benchmark harness.

## TL;DR — which script do I run?

| Script | What it measures | When to run it | Primary audience |
|---|---|---|---|
| `run_eval.py` | Downstream accuracy (MMLU, ARC, HellaSwag, GSM8K, TruthfulQA) | After SFT / post-training — base models at <1B tokens score near random on these | G9 / paper |
| `compute_val_loss.py` | Held-out cross-entropy loss & perplexity | Every milestone checkpoint during pretraining | G1 (data mix), G4 (scaling) |
| `compute_bpb.py` | Bits-per-byte on a frozen text set | Every milestone checkpoint, especially when comparing architectures or tokenizers | G3 (architecture), G4 (scaling), paper |

Perplexity and BPB both come from the same forward pass; they differ in
what they normalise by (tokens vs. UTF-8 bytes). See
[Metric cheatsheet](#metric-cheatsheet) at the bottom.

---

## Current status (Step 1 + 2)

This is an MVP. The two scripts work end-to-end against any HuggingFace
causal LM, but the validation set for `compute_val_loss.py` is still a
small hard-coded text list. Per-domain eval sets (web / code / math /
math_reasoning / textbook) will land in Step 3+4.

| Step | What's shipped | What's pending |
|---|---|---|
| 1 | `compute_val_loss.py` runs on a built-in mini corpus | Per-domain PPL uses placeholder text |
| 2 | `compute_bpb.py` + frozen v1 eval set builder | — |
| 3 | — | `prepare_per_domain_eval.py` |
| 4 | — | `compute_val_loss.py` reads per-domain shards |
| 5 | — | Scale eval sets to ~2M tokens / domain |

---

## Quickstart

Both scripts follow the same conventions as `run_eval.py`: `--model`
accepts an HF name or a local checkpoint directory, `--ckpt_path` can
overlay a raw `.pt` state dict, and results are appended as JSONL.

### Held-out loss / PPL (smoke test)

```bash
python scripts/compute_val_loss.py --model Qwen/Qwen2.5-0.5B --batch_size 2
```

Expected output (order-of-magnitude):

```json
{
  "model": "Qwen/Qwen2.5-0.5B",
  "eval_set": "mini_builtin_v0",
  "nll_per_token": 3.1,
  "ppl": 22.3
}
```

PPL outside `[5, 200]` on Qwen2.5-0.5B usually means something is wrong
(dtype, tokenizer mismatch, OOM silently falling back). NaN / inf →
a broken checkpoint or the loss head.

### BPB on the frozen v1 eval set

1. **Build the eval set once** (downloads ~1.5 MB of text from HF):

   ```bash
   pip install datasets                       # only needed the first time
   python scripts/prepare_bpb_eval.py         # writes data/eval/bpb_v1/
   git add data/eval/bpb_v1/
   git commit -m "Add BPB v1 eval set"
   ```

2. **Score any checkpoint**:

   ```bash
   python scripts/compute_bpb.py --model Qwen/Qwen2.5-0.5B
   ```

   Expected output:

   ```json
   {
     "eval_set_version": "v1",
     "bpb": 0.98,
     "ppl_same_tokenizer": 21.4,
     "bytes_per_token": 3.12
   }
   ```

   BPB outside `[0.5, 2.0]` usually indicates a tokenizer mismatch or a
   corrupted eval set. If the script aborts with `sha256 mismatch`, the
   frozen files were edited — restore them from git or bump to `v2` via
   `prepare_bpb_eval.py --output_dir data/eval/bpb_v2/`.

---

## Integrating with an existing training run

Both scripts read HF-format checkpoints directly, so they work on anything
`pretrain_fineweb.py` produces. A typical scaling-experiment workflow:

```bash
# For every saved checkpoint in a scaling run:
for ckpt in output/<run_name>/checkpoint-*; do
    step=${ckpt##*-}
    python scripts/compute_val_loss.py --model "$ckpt" --step "$step" \
        --output_file results/<run_name>_val_loss.jsonl
    python scripts/compute_bpb.py      --model "$ckpt" --step "$step" \
        --output_file results/<run_name>_bpb.jsonl
done
```

The `--step` flag is optional but makes plotting trivial (each JSONL
record carries its own x-axis).

---

## Notes for each team

**G1 (pretraining data).** Once Step 4 lands, `compute_val_loss.py` will
break out PPL per domain (web / code / math / math_reasoning / textbook).
A math PPL that stays >2× higher than web PPL after 200 M tokens is a
signal to rebalance the mix. Until Step 4, the script only reports a
single overall PPL over a built-in sample — treat it as a pipeline
sanity check, not a data-mix diagnostic.

**G3 (architecture).** Use `compute_bpb.py` for any cross-architecture
comparison. It's tokenizer-independent, so you can compare MoD vs. Hymba
vs. a Qwen-arch baseline without worrying that one tokenizer is "cheating"
by producing fewer tokens per byte.

**G4 (scaling).** The current IsoFLOP plot uses final train loss, which
(a) isn't comparable across tokenizers and (b) can be lowered for free by
under-training (loss is still dropping when a run ends). BPB on the v1
set gives a honest, fixed-axis metric. Suggested: log both PPL and BPB at
every `save_steps` checkpoint, and use BPB as the y-axis in the IsoFLOP
figure.

---

## Eval-set versioning policy

* **`bpb_v1`** — frozen. Any change to the text under `data/eval/bpb_v1/`
  will be caught by the sha256 check in `compute_bpb.py`. If you need
  different text (e.g. to add a new domain), create `bpb_v2` by running
  `prepare_bpb_eval.py --output_dir data/eval/bpb_v2/` and re-run the
  script with `--eval_set_dir data/eval/bpb_v2/`. Paper tables should
  cite the version.
* **`ppl_v1`** (coming in Step 5) — additive changes (extra domains,
  more tokens per domain) bump the patch version; removing or replacing
  content bumps the major version.

---

## Metric cheatsheet

Let `L` be the sum of per-token negative log-likelihoods (in nats) over
`T` predicted tokens of text totalling `B` UTF-8 bytes:

```
nll_per_token = L / T
perplexity    = exp(L / T)                       # tokenizer-dependent
bpb           = L / (ln(2) * B)                  # tokenizer-independent
bytes_per_token = B / T                          # compression ratio
```

The two scripts compute everything from the same three running totals
(`sum_nll_nats`, `total_pred_tokens`, `total_bytes_utf8`) in
`eval_utils.score_texts()`, so numbers from the two scripts on the same
text are internally consistent.

## Downstream-benchmark soft metrics (Delphi)

Hard benchmark scores are at chance for 50M–100M models, but the
underlying **soft** metric scales smoothly with compute. The scaling-law
plotter (`scripts/plot_benchmark_scaling.py`) consumes a JSONL of per-run
eval results with this schema:

```json
{"run_id": "...", "arch": "...", "params": 1e8, "tokens": 1e9, "flops": 6e17,
 "benchmark": "mmlu", "soft": 0.42, "hard": 0.25}
```

### Soft metric definitions

* **Log-prob** family (MMLU, HellaSwag, ARC, PIQA, BBH-MC, MuSR, LogiQA,
  GPQA-Diamond, MC-converted AgentBench/WebArena): mean
  `log P(correct option) − log Σ_i P(option_i)` across examples.
* **Bits-per-byte** family (GSM8K, HumanEval): BPB on the reference
  completion. Lower = better; sign-flip when computing the sigmoid.

### Soft → hard mapping

A sigmoid `hard = 1 / (1 + exp(-k·(soft − s₀)))` is fit on the external
reference pool in `evals/reference_models.json` (Llama / Qwen / OLMo at
multiple sizes). Edit that file to add new reference models; the plotter
re-fits automatically. The pool deliberately spans multiple training
recipes so the sigmoid is not dominated by one family.

### MC conversion for agentic benchmarks

AgentBench and WebArena trajectories are converted to MC by
`evals/agentbench_mc.py`. Each step becomes an MC example with the
expert action at option 0 and `K` distractor actions sampled from the
trajectory's neighborhood. The soft metric is then log-prob of the
expert action; the hard metric is per-step argmax accuracy.

See `docs/scaling_experiments.md § Downstream-benchmark scaling` for how
the plotter ties this back to the IsoFLOP analysis.
