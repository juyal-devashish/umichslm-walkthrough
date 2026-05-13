#!/bin/bash
#SBATCH --job-name=titan-baseline-lr3e4
#SBATCH --output=/shared/4/projects/umich-slm/slurm/titan-pretrain-%j.out
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:A100:1

# Environment setup (mirrors local-slurm/small_baseline_lr3e4_fineweb-1B.sh).
if [[ $(hostname) == "lackerli" ]]; then
  export CUDA_HOME=/usr/local/cuda-12.6
  export PATH=/usr/local/cuda-12.6/bin:/opt/anaconda/bin:/usr/sbin:/usr/bin:/sbin:/bin
  export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
  export TORCH_CUDA_ARCH_LIST="8.0"
fi

if [[ $(hostname) == "sushi" ]]; then
  export NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_NCCL_BLOCKING_WAIT=1
  export NCCL_IB_DISABLE=1
  export NCCL_P2P_LEVEL=NVL
fi

# Number of GPUs on this node. Adjust along with --gres above when scaling out.
NPROC=${SLURM_GPUS_ON_NODE:-1}

# Tokenizer / data: customize for the actual cluster paths.
DISABLE_VERSION_CHECK=1 \
  HF_HOME=/shared/4/models/ \
  HF_HUB_CACHE=/shared/4/models/ \
  HUGGINGFACE_CACHE_DIR=/shared/4/models/ \
  WANDB_PROJECT=umich-slm-titan \
  torchrun --standalone --nproc_per_node=${NPROC} \
    scripts/torchtitan_train.py \
    configs/torchtitan/umichslm_pretrain_baseline.toml
