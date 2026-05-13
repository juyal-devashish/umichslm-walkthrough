#!/bin/bash
#SBATCH --job-name=install-flash-attn
#SBATCH --output=slurm-logs/install-flash-attn-%j.out
#SBATCH --error=slurm-logs/install-flash-attn-%j.err
## --- edit for your site ---
#SBATCH --account=cse598s004w26_class
#SBATCH --partition=standard
# Compiling flash-attn is CPU/RAM heavy; GPU optional. Source builds often need 4–12+ h with
# MAX_JOBS=1 and two arch flags; if the scheduler rejects a long --time, use your site's max
# walltime or a compile/long QoS. Optionally set only the arch you need (e.g. 8.6 OR 12.0).
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-umich-slm}"
CUDA_HOME_PATH="${CUDA_HOME_PATH:-/sw/pkgs/arc/cuda/12.8.1}"
GCC_ROOT="${GCC_ROOT:-/sw/pkgs/arc/gcc/13.2.0}"

# A40 (8.6) + RTX PRO 6000 Blackwell (12.0). Override if nvidia-smi shows something else.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;12.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

mkdir -p slurm-logs

echo "=== $(date) host=$(hostname) ==="
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS"

module load gcc
module load cuda

export CUDA_HOME="$CUDA_HOME_PATH"
export PATH="$CUDA_HOME/bin:$PATH"

# conda first — activation prepends $CONDA_PREFIX/bin and can expose broken c++ symlinks
if [[ -f "$HOME/.conda/etc/profile.d/conda.sh" ]]; then
  source "$HOME/.conda/etc/profile.d/conda.sh"
elif [[ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "ERROR: Could not find conda.sh. Activate conda in this script manually."
  exit 1
fi

conda activate "$CONDA_ENV"

# After conda: force HPC toolchains (re-export every time; Ninja/torch use CXX/CUDAHOSTCXX)
export CC="${GCC_ROOT}/bin/gcc"
export CXX="${GCC_ROOT}/bin/g++"
export CUDAHOSTCXX="$CXX"
export CMAKE_C_COMPILER="$CC"
export CMAKE_CXX_COMPILER="$CXX"
# Prepend so `c++` / `g++` resolve to module GCC, not `x86_64-conda-linux-gnu-c++`
export PATH="${GCC_ROOT}/bin:${CUDA_HOME}/bin:${PATH}"

echo "nvcc: $(command -v nvcc)"
nvcc --version
echo "CC=$CC CXX=$CXX"
echo "which g++: $(command -v g++)  which c++: $(command -v c++)"
"$CXX" --version
if command -v x86_64-conda-linux-gnu-c++ &>/dev/null; then
  echo "NOTE: conda cross-c++ still on PATH; CC/CXX should still override for torch if set (see next lines)."
fi

python -c "import os, pathlib; c=os.environ.get('CC',''); assert c and pathlib.Path(c).exists(), c; print('CC ok:', c)"

pip install --upgrade pip
# Subprocesses must see CC/CXX; explicit env in case a tool drops inherited exports
CC="$CC" CXX="$CXX" CUDAHOSTCXX="$CUDAHOSTCXX" \
  CMAKE_C_COMPILER="$CC" CMAKE_CXX_COMPILER="$CXX" \
  MAX_JOBS="$MAX_JOBS" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  pip install "flash-attn>=2.5.0" --no-build-isolation

python -c "import flash_attn; print('flash_attn OK', getattr(flash_attn, '__version__', ''))"

echo "=== done $(date) ==="
