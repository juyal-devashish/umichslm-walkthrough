# SLURM Workflows for Pretraining and SFT

This guide explains how collaborators should launch UMichSLM pretraining and
SFT jobs on Great Lakes using the generated SLURM workflow scripts.

The important idea is simple: edit a YAML config, generate a `.sh` batch script,
review it, then submit it with `sbatch`.

## Files to Know

| Purpose | File |
|---|---|
| Shared SLURM rendering helpers | `scripts/slurm_common.py` |
| Pretraining generator | `scripts/generate_pretrain_slurm.py` |
| Pretraining config | `configs/slurm/pretrain_greatlakes.yaml` |
| SFT generator | `scripts/generate_sft_slurm.py` |
| SFT config | `configs/slurm/sft_greatlakes.yaml` |
| Generated scripts | `generated-slurm/` |

`generated-slurm/` is intentionally ignored by git. Generated scripts are local
artifacts, not source files.

## Pretraining

Render the default Great Lakes pretraining batch script:

```bash
python scripts/generate_pretrain_slurm.py \
  --config configs/slurm/pretrain_greatlakes.yaml
```

This writes:

```bash
generated-slurm/pretrain/pretrain_workflow.sh
```

Review it:

```bash
less generated-slurm/pretrain/pretrain_workflow.sh
```

Submit it:

```bash
sbatch generated-slurm/pretrain/pretrain_workflow.sh
```

To inspect the generated script without writing it:

```bash
python scripts/generate_pretrain_slurm.py \
  --config configs/slurm/pretrain_greatlakes.yaml \
  --dry_run
```

The default config logs to Weights & Biases with:

```text
project: umich-slm
entity:  umich-slm
```

The generated SLURM scripts require `WANDB_API_KEY` to be available in the
submit environment. Do not put the actual key in git-tracked config files.
They also load the Great Lakes CUDA module `cuda/12.1` before activating the
`umich-slm` conda environment.

Set it before submitting:

```bash
export WANDB_API_KEY=your_key_here
sbatch generated-slurm/pretrain/pretrain_workflow.sh
```

If the key is missing, the generated script exits early with a clear error
before training starts.

## SFT

Prepare SFT data and the extended tokenizer first:

```bash
sbatch local-slurm/prepare_sft_data.sh
sbatch local-slurm/extend_tokenizer.sh
```

Render the default SFT learning-rate sweep:

```bash
python scripts/generate_sft_slurm.py \
  --config configs/slurm/sft_greatlakes.yaml
```

This writes:

```bash
generated-slurm/sft/sft_workflow.sh
```

Review it:

```bash
less generated-slurm/sft/sft_workflow.sh
```

Submit it:

```bash
sbatch generated-slurm/sft/sft_workflow.sh
```

The SFT script calls `scripts/run_sft.py`, which creates a resolved LLaMA
Factory config under:

```bash
generated-slurm/sft-configs/
```

That generated config is also ignored by git.

The SFT workflow also requires `WANDB_API_KEY` in the submit environment:

```bash
export WANDB_API_KEY=your_key_here
sbatch generated-slurm/sft/sft_workflow.sh
```

## Sweeps

A sweep means launching the same kind of run multiple times while changing one
or more settings.

For example, the default SFT config includes:

```yaml
sweep:
  learning_rate: [1.0e-5, 2.0e-5, 5.0e-5]
```

The generator turns this into a SLURM array:

```bash
#SBATCH --array=0-2
```

Each array task gets a different learning rate:

```text
task 0 -> 1.0e-5
task 1 -> 2.0e-5
task 2 -> 5.0e-5
```

You can sweep multiple fields. The generator runs every combination:

```yaml
sweep:
  size: [qwen3_like_tiny, qwen3_like_xsmall]
  target_tokens_b: [0.02, 0.04]
  learning_rate: [3.0e-4, 1.0e-4]
```

That creates 8 jobs: `2 sizes x 2 token budgets x 2 learning rates`.

## Common Edits

To choose a conda environment by name, use:

```yaml
slurm:
  conda_env_name: umich-slm
```

To choose a conda environment by absolute path instead, use:

```yaml
slurm:
  conda_env_path: /scratch/.../envs/umich-slm
```

Use only one of `conda_env_name` or `conda_env_path`.

For pretraining, collaborators usually edit:

```yaml
training:
  size: qwen3_like_small
  target_tokens_b: 1.0
  batch_size: 16
  gradient_accumulation_steps: 8
  learning_rate: 3.0e-4
```

and:

```yaml
data:
  dataset: ...
  dataset_dir: ...
  eval_dataset: ...
```

For SFT, collaborators usually edit:

```yaml
sft:
  model_name_or_path: ...
  dataset: umichslm_sft_train
  eval_dataset: umichslm_sft_val
  learning_rate: 2.0e-5
```

and the sweep:

```yaml
sweep:
  learning_rate: [1.0e-5, 2.0e-5, 5.0e-5]
```

## Checking Jobs

Show your queued/running jobs:

```bash
squeue -u $USER
```

Inspect recent logs:

```bash
ls -lh slurm-logs
tail -f slurm-logs/<log-file>
```

Generated scripts also write SLURM logs under:

```bash
/home/%u/umich-slm/slurm-logs
```

where `%u` is expanded by SLURM to the submitting user.

## Output Locations

The default configs write model outputs to shared scratch:

```bash
/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data/results/
```

Pretraining outputs go under:

```bash
.../results/pre-train/
```

SFT outputs go under:

```bash
.../results/sft/
```

Each run gets a unique slug containing key settings like model size, token
budget, dataset, and learning rate.

## Practical Workflow

Use this loop for new experiments:

1. Edit `configs/slurm/pretrain_greatlakes.yaml` or `configs/slurm/sft_greatlakes.yaml`.
2. Run the matching generator with `--dry_run`.
3. Render the script without `--dry_run`.
4. Review the generated `.sh` file.
5. Submit with `sbatch`.
6. Watch `squeue`, Slurm logs, and WandB.

Do not manually edit files under `generated-slurm/` unless you are debugging a
single local run. Put reusable changes back into the YAML config or generator.
