#!/usr/bin/env bash
#SBATCH --job-name=mpower_alignment
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
set -euo pipefail
umask 077
cd "${MPOWER_CODE_ROOT:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$MPOWER_ADAPT_ROOT/dependencies:$PWD:$PWD/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"${MPOWER_BENCHMARK_PYTHON:?}" -u scripts/prepare_alignment.py
"$MPOWER_BENCHMARK_PYTHON" -u scripts/train_alignment.py
