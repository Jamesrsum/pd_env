#!/usr/bin/env bash
#SBATCH --job-name=mpower_controlled_ast_branch
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
set -euo pipefail
umask 077
cd "${MPOWER_CONTROLLED_CODE:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
policies=(frozen last2 full)
"${MPOWER_BENCHMARK_PYTHON:?}" -u scripts/run_controlled_adaptation.py --stage branch --policy "${policies[${SLURM_ARRAY_TASK_ID:?}]}"
