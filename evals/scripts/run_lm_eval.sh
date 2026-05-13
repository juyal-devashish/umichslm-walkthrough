#!/bin/bash
#SBATCH --job-name=lm-eval-baselines
#SBATCH --mail-user=<ACCOUNT>@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=4:00:00
#SBATCH --account=cse598s004w26_class
#SBATCH --partition=gpu-rtx6000
#SBATCH --gpus=1
#SBATCH --output=/home/%u/umich-slm/slurm-logs/%x-%j.log

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate umich-slm

export PROJECT_DIR="$HOME/umich-slm"
export SCRATCH="/scratch/cse598s004w26_class_root/cse598s004w26_class/shared_data"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export DISABLE_VERSION_CHECK=1

mkdir -p "$PROJECT_DIR/slurm-logs"

cd "$PROJECT_DIR"

RESULTS_DIR="$SCRATCH/results/$USER/lm-eval"

MODELS=(
    "Qwen/Qwen3-0.6B"
)

for MODEL in "${MODELS[@]}"; do
    echo "=============================="
    echo "Evaluating: $MODEL"
    echo "Start: $(date)"
    echo "=============================="

    python evals/scripts/run_general.py \
        --model "$MODEL" \
        --device cuda \
        --batch_size auto \
        --output_path "$RESULTS_DIR" \
        --wandb

    echo "Done: $(date)"
done

echo ""
echo "All evaluations complete. Results in: $RESULTS_DIR"
