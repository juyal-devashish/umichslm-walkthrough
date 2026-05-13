#!/bin/bash
#SBATCH --job-name=titan-baseline-ddp8
#SBATCH --output=/shared/4/projects/umich-slm/slurm/titan-pretrain-%j.out
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:A100:8

# 8-GPU DDP variant. Effective batch = 128 (per_device=8, grad_accum=2, world_size=8).
# This is typically NCCL-bandwidth-bound at this batch size; expect tokens/sec to
# scale sub-linearly past 4 GPUs on the same node. Use the throughput_compare
# script to quantify.

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

NPROC=${SLURM_GPUS_ON_NODE:-8}

DISABLE_VERSION_CHECK=1 \
  HF_HOME=/shared/4/models/ \
  HF_HUB_CACHE=/shared/4/models/ \
  HUGGINGFACE_CACHE_DIR=/shared/4/models/ \
  WANDB_PROJECT=umich-slm-titan \
  torchrun --standalone --nproc_per_node=${NPROC} \
    scripts/torchtitan_train.py \
    configs/torchtitan/umichslm_pretrain_baseline.toml \
    --set training.gradient_accumulation_steps=2 \
    --set parallelism.dp_replicate_degree=8 \
    --set job.run_name=titan_small_baseline_lr3e4_ddp8 \
    --set job.output_dir=./outputs/titan/small_baseline_lr3e4_ddp8 \
    --set logging.wandb_tags=benchmark_v1,baseline,ddp8
