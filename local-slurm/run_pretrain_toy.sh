#!/bin/bash
#SBATCH --job-name=pretrain-toy
#SBATCH --mail-user=[Yourmail]
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=cse598s004w26_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --output=/home/%u/umich-slm/slurm-logs/%x-%j.log

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate umich-slm

export PROJECT_DIR="$HOME/umich-slm"
export SCRATCH="/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export DISABLE_VERSION_CHECK=1

cd "$PROJECT_DIR"
mkdir -p slurm-logs "$SCRATCH/results/pre-train"

# Edit these for your run.
SIZE="qwen3_like_small"
RUN_TAG="toy1b-${SIZE}"
DATA_DIR="$SCRATCH/toy_datasets/1B"
TRAIN_DATASET="mix_code10_math5_math_reasoning5_textbook20_web60_train_1000M"
EVAL_DATASET="mix_code10_math5_math_reasoning5_textbook20_web60_val_30M"
OUTPUT_DIR="$SCRATCH/results/pre-train/$RUN_TAG"
WANDB_PROJECT="umich-slm"
WANDB_ENTITY="umich-slm"

echo "=== UMichSLM Toy 1B Pretraining ==="
echo "Run tag: ${RUN_TAG}"
echo "Preset: ${SIZE}"
echo "Dataset dir: ${DATA_DIR}"
echo "Train dataset: ${TRAIN_DATASET}"
echo "Eval dataset: ${EVAL_DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "Start: $(date)"
echo "==================================="

python scripts/pretrain_fineweb.py \
  --size "$SIZE" \
  --dataset "$TRAIN_DATASET" \
  --dataset_dir "$DATA_DIR" \
  --do_eval \
  --eval_dataset "$EVAL_DATASET" \
  --eval_steps 200 \
  --output_dir "$OUTPUT_DIR" \
  --save_steps 200 \
  --logging_steps 50 \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY"

echo "=== Done: $(date) ==="
