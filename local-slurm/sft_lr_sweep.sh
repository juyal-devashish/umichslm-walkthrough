#!/bin/bash
#SBATCH --job-name=sft-lr-sweep
#SBATCH --mail-user=<ACCOUNT>@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=8:00:00
#SBATCH --account=cse598s004w26_class
#SBATCH --partition=gpu-rtx6000
#SBATCH --gpus=1
#SBATCH --output=/home/%u/umich-slm/slurm-logs/%x-%A_%a.log
#SBATCH --array=0-2

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate umich-slm

export PROJECT_DIR="$HOME/umich-slm"
export SCRATCH="/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$SCRATCH/.cache/huggingface"
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=umichslm-sft
export WANDB_ENTITY=umich-slm

cd "$PROJECT_DIR"
mkdir -p slurm-logs

LRS=(1e-5 2e-5 5e-5)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

TAG="lr${LR}"
OUT_DIR="$SCRATCH/results/$USER/umichslm-sft-${TAG}"
mkdir -p "$OUT_DIR"

echo "=== SFT LR sweep: task=$SLURM_ARRAY_TASK_ID lr=$LR ==="
echo "Output: $OUT_DIR"
echo "Start:  $(date)"

llamafactory-cli train configs/llama_factory/umichslm_sft.yaml \
    learning_rate="$LR" \
    output_dir="$OUT_DIR" \
    logging_dir="$OUT_DIR/tb" \
    run_name="umichslm-sft-${TAG}"

echo "=== Done: $(date) ==="
