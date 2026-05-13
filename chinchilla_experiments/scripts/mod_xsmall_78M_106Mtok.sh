#!/bin/bash
#SBATCH --job-name=scale-mod_xsmall_78M_106Mtok
#SBATCH --output=/home/%u/umich-slm/slurm-logs/mod_xsmall_78M_106Mtok-%j.out
#SBATCH --error=/home/%u/umich-slm/slurm-logs/mod_xsmall_78M_106Mtok-%j.err
#SBATCH --partition=spgpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --account=cse598s004w26_class
#SBATCH --mail-user=vnagpal@umich.edu
#SBATCH --mail-type=END,FAIL
# Scaling Law Experiment: mod_xsmall_78M_106Mtok
# Model: mod_xsmall (78.3M params)
# Training tokens: 106M
# Estimated FLOPs: 5.00e+16

echo "========================================"
echo "Experiment: mod_xsmall_78M_106Mtok"
echo "Model:      mod_xsmall (78.3M params)"
echo "Tokens:     106M"
echo "FLOPs:      5.00e+16"
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
  --output_dir /scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data/vnagpal/results/mod_xsmall_78M_106Mtok/ \
  --size mod_xsmall \
  --batch_size 24 \
  --gradient_accumulation_steps 5 \
  --learning_rate 3.79e-04 \
  --target_tokens_b 0.106 \
  --val_percent 0 \
  --dataset mix_code10_math5_math_reasoning5_textbook20_web60_train_200M \
  --dataset_dir /scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data/toy_datasets/200M \
  --do_eval --eval_dataset mix_code10_math5_math_reasoning5_textbook20_web60_val_6M --eval_steps 200

EXIT_CODE=$?
echo "========================================"
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "========================================"
exit $EXIT_CODE
