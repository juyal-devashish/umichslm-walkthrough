#!/bin/bash
#SBATCH --job-name=scale-qwen3_like_tiny_5M_20Mtok
#SBATCH --output=/home/%u/umich-slm/slurm-logs/%x-%j.log/scaling-%j.out
#SBATCH --partition=gpu-rtx6000
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --account=cse598s004w26_class
# Scaling Law Experiment: qwen3_like_tiny_5M_20Mtok
# Model: qwen3_like_tiny (5.0M params)
# Training tokens: 20M
# Estimated FLOPs: 5.98e+14

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

HF_HOME=/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data/.cache/huggingface \
  python scripts/pretrain_fineweb.py \
  --output_dir /scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data/results/qwen3_like_tiny_5M_20Mtok/ \
  --size qwen3_like_tiny \
  --batch_size 32 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1.00e-03 \
  --target_tokens_b 0.020 \
  --dataset mix_code10_math5_math_reasoning5_textbook20_web60_train_20M \
  --dataset_dir /scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data/toy_datasets/20M \
  --do_eval --eval_dataset mix_code10_math5_math_reasoning5_textbook20_web60_val_0.6M --eval_steps 200
