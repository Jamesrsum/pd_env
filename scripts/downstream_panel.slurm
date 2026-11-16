#!/usr/bin/env bash
#SBATCH --job-name=mpower_downstream_panel
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
set -euo pipefail
umask 077
cd "${MPOWER_DOWNSTREAM_CODE:?}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"${MPOWER_BENCHMARK_PYTHON:?}" -u scripts/run_downstream_panel.py "$@"
