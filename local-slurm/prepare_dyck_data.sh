#!/bin/bash
#SBATCH --job-name=pre-pretrain
#SBATCH --mail-user=<ACCOUNT>@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=cse598s004w26_class
#SBATCH --partition=standard
#SBATCH --output=/home/%u/umich-slm/slurm-logs/%x-%j.log
#SBATCH --error=/home/%u/umich-slm/slurm-logs/%x-%j.err

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate umich-slm

# === Paths ===
export PROJECT_DIR="$HOME/umich-slm"
export SCRATCH="/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TOKENIZER_PATH="HuggingFaceTB/SmolLM2-1.7B-Instruct"

cd "$PROJECT_DIR"
mkdir -p "$SCRATCH/dyck_data" slurm-logs

echo "=== Preparing Shuffle-Dyck-k dataset ==="
echo "Output: $SCRATCH/dyck_data"
echo "Start: $(date)"

python utils_dyck.py \
  --tokenizer $TOKENIZER_PATH \
  --output_dir "$SCRATCH/dyck_data" \
  --save_arrow

echo "=== Done: $(date) ==="