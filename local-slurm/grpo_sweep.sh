#!/bin/bash
#SBATCH --job-name=grpo-sweep
#SBATCH --mail-user=<ACCOUNT>@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=10:00:00
#SBATCH --account=cse598s004w26_class
#SBATCH --partition=gpu-rtx6000
#SBATCH --gpus=1
#SBATCH --output=/home/%u/umich-slm/slurm-logs/%x-%A_%a.log
#SBATCH --array=0-3

set -euo pipefail

# Required: path to the winning SFT checkpoint.
# Usage:
#   SFT_WINNER=/scratch/.../umichslm-sft-lr2e-5 sbatch local-slurm/grpo_sweep.sh
: "${SFT_WINNER:?set SFT_WINNER=<path to best SFT checkpoint>}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate umich-slm

export PROJECT_DIR="$HOME/umich-slm"
export SCRATCH="/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$SCRATCH/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=umichslm-grpo
export WANDB_ENTITY=umich-slm

cd "$PROJECT_DIR"
mkdir -p slurm-logs

# Sweep over num_generations × epsilon (2×2 grid)
NUM_GENS=(4  4  8  8)
EPSILONS=(0.1 0.2 0.1 0.2)

G=${NUM_GENS[$SLURM_ARRAY_TASK_ID]}
EPS=${EPSILONS[$SLURM_ARRAY_TASK_ID]}

TAG="g${G}-eps${EPS}"
OUT_DIR="$SCRATCH/results/$USER/umichslm-grpo-${TAG}"
mkdir -p "$OUT_DIR"

TRAIN_DATA="$SCRATCH/grpo_data/v1/gsm8k/train.jsonl"
VAL_DATA="$SCRATCH/grpo_data/v1/gsm8k/val.jsonl"

echo "=== GRPO sweep: task=$SLURM_ARRAY_TASK_ID G=$G eps=$EPS ==="
echo "SFT winner: $SFT_WINNER"
echo "Output:     $OUT_DIR"
echo "Start:      $(date)"

python scripts/run_grpo.py \
    --config     configs/llama_factory/umichslm_grpo.yaml \
    --model_path "$SFT_WINNER" \
    --train_data "$TRAIN_DATA" \
    --val_data   "$VAL_DATA" \
    --output_dir "$OUT_DIR" \
    --num_generations "$G" \
    --epsilon "$EPS" \
    --run_name "umichslm-grpo-${TAG}"

echo "=== Done: $(date) ==="
