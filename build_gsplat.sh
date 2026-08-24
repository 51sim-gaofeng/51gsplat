#!/usr/bin/env bash
set -euo pipefail

# Build and install gsplat into conda env "4dgs" on Ubuntu/Linux.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

# Enable `conda activate` in non-interactive shells.
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "Activating conda environment: 4dgs"
conda activate 4dgs

# Compiler/runtime tuning knobs (can be overridden by exported env vars).
: "${NVCC_THREADS:=5}"
: "${NVCC_SPLIT_COMPILE:=8}"
# 32 matches padded semantic raster channels default in this project.
: "${NUM_CHANNELS:=1,2,3,4,32}"
export NVCC_THREADS NVCC_SPLIT_COMPILE NUM_CHANNELS

# Parallel build jobs for ninja / setuptools.
: "${MAX_JOBS:=$(nproc)}"
export MAX_JOBS

# Derive CUDA_HOME from nvcc path when not explicitly set.
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  NVCC_BIN="$(command -v nvcc)"
  export CUDA_HOME="$(cd "$(dirname "${NVCC_BIN}")/.." && pwd)"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDA_PATH="${CUDA_HOME}"
fi

echo "Toolchain check..."
command -v python >/dev/null 2>&1 || { echo "ERROR: python not found" >&2; exit 1; }
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found" >&2; exit 1; }
command -v ninja >/dev/null 2>&1 || { echo "ERROR: ninja not found" >&2; exit 1; }

echo "Building and installing gsplat from source into 4dgs env: ${SCRIPT_DIR}"
python -m pip install "${SCRIPT_DIR}" --no-build-isolation -v

echo "Build completed."
