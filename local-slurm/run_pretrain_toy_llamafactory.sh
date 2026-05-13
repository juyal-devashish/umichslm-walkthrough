#!/bin/bash
#SBATCH --job-name=pretrain-l40
#SBATCH --mail-user=[Yourmail]
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=cse598s004w26_class
#SBATCH --partition=gpu-rtx6000
#SBATCH --gpus=1
#SBATCH --output=/home/%u/CSE598_SLM/umich-slm/slurm-logs/%x-%j.log

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate umich-slm

export PROJECT_DIR="$HOME/CSE598_SLM/umich-slm"
export SCRATCH="/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export DISABLE_VERSION_CHECK=1
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH}"
export TOKENIZER_PATH="$HF_HOME/hub/models--HuggingFaceTB--SmolLM2-1.7B-Instruct/snapshots/31b70e2e869a7173562077fd711b654946d38674"
export WANDB_PROJECT="umich-slm"
export WANDB_ENTITY="liuzoro4239-liu"
# export WANDB_MODE=offline

cd "$PROJECT_DIR"
mkdir -p slurm-logs outputs

# Keep these aligned with configs/llama_factory/umichslm_pretrain.yaml.
SIZE="qwen3_like_small"
BOOTSTRAP_ROOT="$PROJECT_DIR/outputs/bootstrap"
BOOTSTRAP_RUN_NAME="toy1b_qwen3_like_small_init"
DATA_DIR="$SCRATCH/toy_datasets/1B"
TRAIN_DATASET="mix_code10_math5_math_reasoning5_textbook20_web60_train_1000M"
EVAL_DATASET="mix_code10_math5_math_reasoning5_textbook20_web60_val_30M"
CONFIG_PATH="configs/llama_factory/umichslm_pretrain.yaml"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
GPU_LABEL="${SLURM_JOB_PARTITION:-local}"
RUN_BASENAME="umichslm_pretrain_toy_1b_${SIZE}_${GPU_LABEL}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  RUN_NAME="${RUN_BASENAME}_${TIMESTAMP}_job${SLURM_JOB_ID}"
else
  RUN_NAME="${RUN_BASENAME}_${TIMESTAMP}"
fi
OUTPUT_DIR="$PROJECT_DIR/outputs/$RUN_NAME"
LOGGING_DIR="$OUTPUT_DIR/logs"

export WANDB_NAME="$RUN_NAME"


echo "=== UMichSLM Toy 1B Pretraining via LLaMA Factory ==="
echo "Preset: ${SIZE}"
echo "Bootstrap root: ${BOOTSTRAP_ROOT}"
echo "Config: ${CONFIG_PATH}"
echo "Dataset dir: ${DATA_DIR}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Run name: ${RUN_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Logging dir: ${LOGGING_DIR}"
echo "W&B project: ${WANDB_PROJECT}"
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
