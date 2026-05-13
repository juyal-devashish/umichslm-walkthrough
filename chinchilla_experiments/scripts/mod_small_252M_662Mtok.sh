#!/bin/bash
#SBATCH --job-name=scale-mod_small_252M_662Mtok
#SBATCH --output=/home/%u/umich-slm/slurm-logs/mod_small_252M_662Mtok-%j.out
#SBATCH --error=/home/%u/umich-slm/slurm-logs/mod_small_252M_662Mtok-%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --account=cse598s004w26_class
#SBATCH --mail-user=vnagpal@umich.edu
#SBATCH --mail-type=END,FAIL
# Scaling Law Experiment: mod_small_252M_662Mtok
# Model: mod_small (251.7M params)
# Training tokens: 662M
# Estimated FLOPs: 1.00e+18

echo "========================================"
echo "Experiment: mod_small_252M_662Mtok"
echo "Model:      mod_small (251.7M params)"
echo "Tokens:     662M"
echo "FLOPs:      1.00e+18"
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

HF_HOME=/scratch/cse598s004w26_class_root/cse598s004w26_class/vnagpal/.cache/huggingface \
  TOKENIZERS_PARALLELISM=false \
  python scripts/pretrain_fineweb.py \
  --output_dir /scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data/vnagpal/results/mod_small_252M_662Mtok/ \
  --size mod_small \
  --batch_size 16 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2.11e-04 \
  --target_tokens_b 0.662 \
  --val_percent 0 \
  --dataset mix_code10_math5_math_reasoning5_textbook20_web60_train_1000M \
  --dataset_dir /scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data/toy_datasets/1B \
  --do_eval --eval_dataset mix_code10_math5_math_reasoning5_textbook20_web60_val_30M --eval_steps 200

EXIT_CODE=$?
echo "========================================"
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "========================================"
exit $EXIT_CODE
