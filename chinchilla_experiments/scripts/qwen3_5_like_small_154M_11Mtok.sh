#!/bin/bash
#SBATCH --job-name=scale-qwen3_5_like_small_154M_11Mtok
#SBATCH --output=/home/%u/umich-slm/slurm-logs/qwen3_5_like_small_154M_11Mtok-%j.out
#SBATCH --error=/home/%u/umich-slm/slurm-logs/qwen3_5_like_small_154M_11Mtok-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --account=<account>
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=END,FAIL
# Scaling Law Experiment: qwen3_5_like_small_154M_11Mtok
# Model: qwen3_5_like_small (154.0M params)
# Training tokens: 11M
# Estimated FLOPs: 1.00e+16

echo "========================================"
echo "Experiment: qwen3_5_like_small_154M_11Mtok"
echo "Model:      qwen3_5_like_small (154.0M params)"
echo "Tokens:     11M"
echo "FLOPs:      1.00e+16"
echo "Host:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:      $(date)"
echo "Job ID:     $SLURM_JOB_ID"
echo "========================================"

# Initialize and activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate umich-slm

# Project directory
export PROJECT_DIR="$HOME/umich-slm"
cd "$PROJECT_DIR"

# Host-specific environment setup
if [[ $(hostname) == "lackerli"* ]]; then
  export CUDA_HOME=/usr/local/cuda-12.6
  export PATH=/usr/local/cuda-12.6/bin:/opt/anaconda/bin:/usr/sbin:/usr/bin:/sbin:/bin
  export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
  export TORCH_CUDA_ARCH_LIST="8.0"
fi

if [[ $(hostname) == "sushi"* ]]; then
  export NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_NCCL_BLOCKING_WAIT=1
  export NCCL_IB_DISABLE=1
  export NCCL_P2P_LEVEL=NVL
fi

HF_HOME=/path/to/data \
  TOKENIZERS_PARALLELISM=false \
  python scripts/pretrain_fineweb.py \
  --output_dir /path/to/data \
  --size qwen3_5_like_small \
  --batch_size 16 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2.70e-04 \
  --target_tokens_b 0.011 \
  --val_percent 0 \
  --dataset mix_code10_math5_math_reasoning5_textbook20_web60_train_20M \
  --dataset_dir /path/to/data \
  --do_eval --eval_dataset mix_code10_math5_math_reasoning5_textbook20_web60_val_0.6M --eval_steps 200

EXIT_CODE=$?
echo "========================================"
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "========================================"
exit $EXIT_CODE
