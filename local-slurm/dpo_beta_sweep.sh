#!/bin/bash
#SBATCH --job-name=dpo-beta-sweep
#SBATCH --mail-user=<ACCOUNT>@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=6:00:00
#SBATCH --account=cse598s004w26_class
#SBATCH --partition=gpu-rtx6000
#SBATCH --gpus=1
#SBATCH --output=/home/%u/umich-slm/slurm-logs/%x-%A_%a.log
#SBATCH --array=0-1

set -euo pipefail

# Required: path to the winning SFT checkpoint (e.g. the best LR dir).
# Pass at submit time:
#   SFT_WINNER=/scratch/.../umichslm-sft-lr2e-5 sbatch local-slurm/dpo_beta_sweep.sh
: "${SFT_WINNER:?set SFT_WINNER=<path to best SFT checkpoint>}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate umich-slm

export PROJECT_DIR="$HOME/umich-slm"
export SCRATCH="/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$SCRATCH/.cache/huggingface"
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=umichslm-dpo
export WANDB_ENTITY=umich-slm

cd "$PROJECT_DIR"
mkdir -p slurm-logs

BETAS=(0.1 0.5)
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

TAG="beta${BETA}"
OUT_DIR="$SCRATCH/results/$USER/umichslm-dpo-${TAG}"
mkdir -p "$OUT_DIR"

echo "=== DPO β sweep: task=$SLURM_ARRAY_TASK_ID beta=$BETA ==="
echo "SFT winner: $SFT_WINNER"
echo "Output:     $OUT_DIR"
echo "Start:      $(date)"

llamafactory-cli train configs/llama_factory/umichslm_dpo_80k.yaml \
    model_name_or_path="$SFT_WINNER" \
    ref_model="$SFT_WINNER" \
    dpo_beta="$BETA" \
    output_dir="$OUT_DIR" \
    logging_dir="$OUT_DIR/tb" \
    run_name="umichslm-dpo-${TAG}"

echo "=== Done: $(date) ==="
