#!/bin/bash
#SBATCH --job-name=prep-dpo-data
#SBATCH --mail-user=<ACCOUNT>@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --account=cse598s004w26_class
#SBATCH --partition=standard
#SBATCH --output=/home/%u/umich-slm/slurm-logs/%x-%j.log

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate umich-slm

export PROJECT_DIR="$HOME/umich-slm"
export SCRATCH="/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$SCRATCH/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"
mkdir -p slurm-logs

OUT_DIR="$SCRATCH/dpo_data/v1/ultrafeedback"
mkdir -p "$OUT_DIR"

echo "=== Preparing UltraFeedback DPO pairs ==="
echo "Output: $OUT_DIR"
echo "Start:  $(date)"

python scripts/dpo_data/prepare_ultrafeedback.py \
    --output_dir "$OUT_DIR" \
    --n 15000 \
    --seed 42

echo "=== Done: $(date) ==="
ls -lh "$OUT_DIR"
