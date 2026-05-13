#!/bin/bash
#SBATCH --job-name=prep-sft-data
#SBATCH --mail-user=<ACCOUNT>@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=4:00:00
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

RAW_DIR="$SCRATCH/sft_data/v1/raw"
MIX_DIR="$SCRATCH/sft_data/v1/sft_mix_80k"
mkdir -p "$RAW_DIR" "$MIX_DIR"

echo "=== Preparing 80k SFT mix ==="
echo "Raw:   $RAW_DIR"
echo "Mix:   $MIX_DIR"
echo "Start: $(date)"

# Per-source caps targeting 80k total.
python scripts/sft_data/prepare_am_r1.py          --output_dir "$RAW_DIR" --n 21000
python scripts/sft_data/prepare_openr1_math.py    --output_dir "$RAW_DIR" --n 9000
python scripts/sft_data/prepare_openthoughts_code.py --output_dir "$RAW_DIR" --n 10000
python scripts/sft_data/prepare_glaive.py         --output_dir "$RAW_DIR" --n 15000
python scripts/sft_data/prepare_webglm.py         --output_dir "$RAW_DIR" --n 10000
python scripts/sft_data/prepare_smoltalk2.py      --output_dir "$RAW_DIR" --n 15000

python scripts/sft_data/build_sft_mix.py \
    --input_dir "$RAW_DIR" \
    --output_dir "$MIX_DIR" \
    --val_frac 0.03 \
    --seed 42

echo "=== Done: $(date) ==="
ls -lh "$MIX_DIR"
