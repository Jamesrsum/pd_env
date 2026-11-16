#!/usr/bin/env bash
#SBATCH --job-name=mpower_tabular_smoke
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
umask 077
cd "${MPOWER_LITERATURE_CODE:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"${MPOWER_BENCHMARK_PYTHON:?}" -u scripts/run_tabular_literature.py --smoke
