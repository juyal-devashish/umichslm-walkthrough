#!/bin/bash
#SBATCH --job-name=extend-tok
#SBATCH --mail-user=<ACCOUNT>@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0:30:00
#SBATCH --account=cse598s004w26_class
#SBATCH --partition=standard
#SBATCH --output=/home/%u/umich-slm/slurm-logs/%x-%j.log

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate umich-slm

export PROJECT_DIR="$HOME/umich-slm"
export SCRATCH="/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data"

cd "$PROJECT_DIR"

BASE="$SCRATCH/results/qwen3_like_small_101M_1000Mtok/models/fineweb_qwen3_like_small_1.0B_20260410-0114"
OUT="$SCRATCH/results/umichslm-base-6tok"

echo "=== Extending tokenizer ==="
echo "Base: $BASE"
echo "Out:  $OUT"
echo "Start: $(date)"

python scripts/sft_data/extend_tokenizer.py \
    --base_model "$BASE" \
    --output_dir "$OUT"

echo "=== Done: $(date) ==="
ls -lh "$OUT"
