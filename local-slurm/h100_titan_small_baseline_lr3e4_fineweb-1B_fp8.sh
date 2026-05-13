#!/bin/bash
#SBATCH --job-name=titan-h100-baseline-fp8
#SBATCH --output=/shared/4/projects/umich-slm/slurm/titan-pretrain-%j.out
#SBATCH --partition=spgpu
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:H100:1

# H100 fp8 + torch.compile baseline. Adjust --gres count and --partition to
# match the actual cluster's H100 partition (Great Lakes spgpu / spgpu2 etc.).
# Float8 only kicks in on SM>=9.0; apply_float8 will raise on older GPUs so
# runs fail fast rather than silently fall back.

if [[ $(hostname) == "lackerli" ]]; then
  export CUDA_HOME=/usr/local/cuda-12.6
  export PATH=/usr/local/cuda-12.6/bin:/opt/anaconda/bin:/usr/sbin:/usr/bin:/sbin:/bin
  export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
  export TORCH_CUDA_ARCH_LIST="9.0"   # Hopper
fi

if [[ $(hostname) == "sushi" ]]; then
  export NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_NCCL_BLOCKING_WAIT=1
  export NCCL_IB_DISABLE=1
  export NCCL_P2P_LEVEL=NVL
fi

# torch.compile + Triton can spend 5+ minutes on first compile; suppress dynamo
# noise but keep tracebacks for real failures.
export TORCH_LOGS="recompiles_verbose"
export TORCHDYNAMO_VERBOSE=0

NPROC=${SLURM_GPUS_ON_NODE:-1}

DISABLE_VERSION_CHECK=1 \
  HF_HOME=/shared/4/models/ \
  HF_HUB_CACHE=/shared/4/models/ \
  HUGGINGFACE_CACHE_DIR=/shared/4/models/ \
  WANDB_PROJECT=umich-slm-titan \
  torchrun --standalone --nproc_per_node=${NPROC} \
    scripts/torchtitan_train.py \
    configs/torchtitan/umichslm_pretrain_baseline_h100_fp8.toml
