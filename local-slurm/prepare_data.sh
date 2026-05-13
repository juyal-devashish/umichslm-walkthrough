#!/bin/bash
#SBATCH --job-name=prep-data
#SBATCH --mail-user=<ACCOUNT>@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1:00:00
#SBATCH --account=cse598s004w26_class
#SBATCH --partition=gpu_mig40
#SBATCH --gpus=1
#SBATCH --output=/home/%u/umich-slm/slurm-logs/%x-%j.log

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate umich-slm

# === Paths ===
export PROJECT_DIR="$HOME/umich-slm"
export SCRATCH="/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data"
export HF_HOME="$SCRATCH/.cache/huggingface"

cd "$PROJECT_DIR"
mkdir -p "$SCRATCH/fineweb-edu-1B" slurm-logs

echo "=== Preparing FineWeb-Edu dataset ==="
echo "Output: $SCRATCH/fineweb-edu-1B"
echo "Start: $(date)"

python scripts/prepare_fineweb_dataset.py \
  --output_dir "$SCRATCH/fineweb-edu-1B" \
  --train_tokens 20000000 \
  --val_percent 3.0 \
  --no_verify

echo "=== Done: $(date) ==="