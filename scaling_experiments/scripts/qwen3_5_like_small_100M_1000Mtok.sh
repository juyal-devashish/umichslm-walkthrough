#!/bin/bash
#SBATCH --job-name=scale-qwen3_5_like_small_100M_1000Mtok
#SBATCH --output=/home/%u/umich-slm/slurm-logs/%x-%j.log/scaling-%j.out
#SBATCH --partition=gpu-rtx6000
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --account=<account>
# Scaling Law Experiment: qwen3_5_like_small_100M_1000Mtok
# Model: qwen3_5_like_small (99.8M params)
# Training tokens: 1000M
# Estimated FLOPs: 5.99e+17

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
  python scripts/pretrain_fineweb.py \
  --output_dir /path/to/data \
  --size qwen3_5_like_small \
  --batch_size 24 \
  --gradient_accumulation_steps 5 \
  --learning_rate 3.36e-04 \
  --target_tokens_b 1.000 \
  --dataset mix_code10_math5_math_reasoning5_textbook20_web60_train_1000M \
  --dataset_dir /path/to/data \
  --do_eval --eval_dataset mix_code10_math5_math_reasoning5_textbook20_web60_val_30M --eval_steps 200
