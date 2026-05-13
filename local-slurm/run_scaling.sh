#!/bin/bash
#SBATCH --job-name=scaling-exp
#SBATCH --mail-user=<ACCOUNT>@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=2:00:00
#SBATCH --account=cse598s004w26_class
#SBATCH --partition=gpu-rtx6000
#SBATCH --gpus=1
#SBATCH --output=/home/%u/umich-slm/slurm-logs/%x-%j.log

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate umich-slm

# === Paths ===
export PROJECT_DIR="$HOME/umich-slm"
export SCRATCH="/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data"
export HF_HOME="$SCRATCH/.cache/huggingface"
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"
mkdir -p slurm-logs

HIDDEN=256
LAYERS=4
HEADS=4
INTERMEDIATE=682
LR=1e-3
BATCH=64
GRAD_ACCUM=2
TARGET_TOKENS_B=0.050

DATA_DIR="$SCRATCH/fineweb-edu-1B"
OUTPUT_DIR="$SCRATCH/results/$USER/h${HIDDEN}_l${LAYERS}_${TARGET_TOKENS_B}B"

mkdir -p "$OUTPUT_DIR"

echo "=== Scaling Experiment ==="
echo "Model: hidden=${HIDDEN}, layers=${LAYERS}, heads=${HEADS}, mlp=${INTERMEDIATE}"
echo "Tokens: ${TARGET_TOKENS_B}B, LR: ${LR}, Batch: $((BATCH * GRAD_ACCUM))"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Start: $(date)"
echo "=========================="

python scripts/pretrain_fineweb.py \
  --output_dir "$OUTPUT_DIR" \
  --size custom \
  --hidden_size "$HIDDEN" \
  --num_hidden_layers "$LAYERS" \
  --num_attention_heads "$HEADS" \
  --intermediate_size "$INTERMEDIATE" \
  --batch_size "$BATCH" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LR" \
  --target_tokens_b "$TARGET_TOKENS_B" \
  --dataset fineweb_edu_1B_train \
  --dataset_dir "$DATA_DIR" \
  --do_eval --eval_dataset fineweb_edu_1B_val --eval_steps 200 \
  --wandb_project scaling-experiment \
  --wandb_entity umich-slm

echo "=== Done: $(date) ==="