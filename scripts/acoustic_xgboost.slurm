#!/usr/bin/env bash
#SBATCH --job-name=mpower_acoustic_xgboost
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
set -euo pipefail
umask 077
cd "${MPOWER_XGBOOST_CODE:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MPOWER_BENCHMARK_PYTHON="${MPOWER_BENCHMARK_PYTHON:?}"
bash scripts/xgboost_python.sh -u scripts/run_acoustic_xgboost.py "$@"
