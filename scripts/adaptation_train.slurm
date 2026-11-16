#!/usr/bin/env bash
#SBATCH --job-name=mpower_adaptation
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
set -euo pipefail
umask 077
cd "${MPOWER_CODE_ROOT:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$MPOWER_ADAPT_ROOT/dependencies:$PWD:$PWD/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"${MPOWER_BENCHMARK_PYTHON:?}" -u scripts/run_adaptation.py --task "${SLURM_ARRAY_TASK_ID:?}"
