#!/bin/bash
#SBATCH --job-name=pre-pretrain
#SBATCH --mail-user=<ACCOUNT>@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --account=cse598s004w26_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --output=/home/%u/umich-slm/slurm-logs/%x-%j.log
#SBATCH --error=/home/%u/umich-slm/slurm-logs/%x-%j.err

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate umich-slm

export PROJECT_DIR="$HOME/umich-slm"
export SCRATCH="/scratch/cse598s004w26_class_root/cse598s004w26_class/"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export DISABLE_VERSION_CHECK=1
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH}"
export TOKENIZER_PATH="HuggingFaceTB/SmolLM2-1.7B-Instruct"
export WANDB_PROJECT="umich-slm"
export WANDB_API_KEY="" # Replace with your W&B API Key
export WANDB_ENTITY=""  # Replace with your W&B username

cd "$PROJECT_DIR"
mkdir -p slurm-logs outputs

# Keep these aligned with configs/llama_factory/umichslm_prepretrain.yaml.
SIZE="qwen3_like_small"
BOOTSTRAP_ROOT="$PROJECT_DIR/outputs/bootstrap"
BOOTSTRAP_RUN_NAME="umichslm_prepretrain_shuffle_dyck_k"
DATA_DIR=$SCRATCH/shared_data/dyck_data/dyck_64_arrow_len1024
TRAIN_DATASET="shuffle_dyck_k"
CONFIG_PATH="configs/llama_factory/umichslm_prepretrain.yaml"

echo "=== Step 1: Pre-pretraining Model on Shuffle-Dyck-k for Warming Up ==="
echo "Preset: ${SIZE}"
echo "Bootstrap root: ${BOOTSTRAP_ROOT}"
echo "Config: ${CONFIG_PATH}"
echo "Dataset dir: ${DATA_DIR}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Start: $(date)"
echo "====================================================="

# Step 1: create a fresh initialized checkpoint at the exact path used by the YAML.
python scripts/pretrain.py \
  --size "$SIZE" \
  --output_dir "$BOOTSTRAP_ROOT" \
  --run_name "$BOOTSTRAP_RUN_NAME" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --dry_run

# Step 2: run LLaMA Factory through the repo wrapper so UMichSLM is registered.
python scripts/llamafactory_train.py train "$CONFIG_PATH"

echo "=== Done: $(date) ==="
