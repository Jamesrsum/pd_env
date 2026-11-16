#!/usr/bin/env bash
#SBATCH --job-name=mpower_controlled_ast_smoke
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
umask 077
cd "${MPOWER_CONTROLLED_CODE:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"${MPOWER_BENCHMARK_PYTHON:?}" -u scripts/run_controlled_adaptation.py --stage warmup --smoke
for policy in frozen last2 full; do
  "$MPOWER_BENCHMARK_PYTHON" -u scripts/run_controlled_adaptation.py --stage branch --policy "$policy" --smoke
done
