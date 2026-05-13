#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --mail-user=<ACCOUNT>@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=cse598s004w26_class
#SBATCH --partition=gpu-rtx6000
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

# Keep these aligned with configs/llama_factory/umichslm_pretrain_from_checkpoint.yaml.
SIZE="qwen3_like_small"
BOOTSTRAP_ROOT="$PROJECT_DIR/outputs/bootstrap"
BOOTSTRAP_RUN_NAME="umichslm_pretrain_toy_200M_shuffle_dyck_k"
DATA_DIR=$SCRATCH/shared_data/toy_datasets/200M
TRAIN_DATASET="mix_code10_math5_math_reasoning5_textbook20_web60_train_200M"
EVAL_DATASET="mix_code10_math5_math_reasoning5_textbook20_web60_val_6M"
CONFIG_PATH=configs/llama_factory/umichslm_pretrain_from_checkpoint.yaml

echo "=== Step 2: Pretraining using checkpoint from Pre-pretraining ==="
echo "Preset: ${SIZE}"
echo "Bootstrap root: ${BOOTSTRAP_ROOT}"
echo "Config: ${CONFIG_PATH}"
echo "Dataset dir: ${DATA_DIR}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Start: $(date)"
echo "====================================================="

# Step 1: create a fresh initialized checkpoint at the exact path used by the YAML.
# Skipped: not initializing new checkpoint

# Step 2: run LLaMA Factory through the repo wrapper so UMichSLM is registered.
python scripts/llamafactory_train.py train "$CONFIG_PATH"

echo "=== Done: $(date) ==="
