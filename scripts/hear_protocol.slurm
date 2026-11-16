#!/usr/bin/env bash
#SBATCH --job-name=mpower_hear_protocol
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
set -euo pipefail
umask 077
cd "${MPOWER_HEAR_CODE:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
stage="${1:?}"; shift
if [[ "$stage" == smoke ]]; then
    "${MPOWER_BENCHMARK_PYTHON:?}" -u scripts/run_hear_protocol.py --stage extract --smoke
    "${MPOWER_BENCHMARK_PYTHON:?}" -u scripts/run_hear_protocol.py --stage warmup --smoke
    for policy in frozen last2 full; do
        "$MPOWER_BENCHMARK_PYTHON" -u scripts/run_hear_protocol.py --stage branch --policy "$policy" --block-path "${MPOWER_HEAR_BLOCK_PATH:?}" --smoke
    done
else
    "${MPOWER_BENCHMARK_PYTHON:?}" -u scripts/run_hear_protocol.py --stage "$stage" --block-path "${MPOWER_HEAR_BLOCK_PATH:?}" "$@"
fi
