#!/bin/bash
#SBATCH --job-name=scale-hymba_like_tiny_26M_640Mtok
#SBATCH --output=/home/%u/umich-slm/slurm-logs/hymba_like_tiny_26M_640Mtok-%j.out
#SBATCH --error=/home/%u/umich-slm/slurm-logs/hymba_like_tiny_26M_640Mtok-%j.err
#SBATCH --partition=spgpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --account=<account>
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=END,FAIL
# Scaling Law Experiment: hymba_like_tiny_26M_640Mtok
# Model: hymba_like_tiny (26.1M params)
# Training tokens: 640M
# Estimated FLOPs: 1.00e+17

echo "========================================"
echo "Experiment: hymba_like_tiny_26M_640Mtok"
echo "Model:      hymba_like_tiny (26.1M params)"
echo "Tokens:     640M"
echo "FLOPs:      1.00e+17"
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
  --size hymba_like_tiny \
  --batch_size 32 \
  --gradient_accumulation_steps 4 \
  --learning_rate 6.57e-04 \
  --target_tokens_b 0.640 \
  --val_percent 0 \
  --dataset mix_code10_math5_math_reasoning5_textbook20_web60_train_1000M \
  --dataset_dir /path/to/data \
  --do_eval --eval_dataset mix_code10_math5_math_reasoning5_textbook20_web60_val_30M --eval_steps 200

EXIT_CODE=$?
echo "========================================"
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "========================================"
exit $EXIT_CODE
