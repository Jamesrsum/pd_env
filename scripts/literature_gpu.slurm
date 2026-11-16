#!/usr/bin/env bash
#SBATCH --job-name=mpower_literature
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
set -euo pipefail
umask 077
cd "${MPOWER_LITERATURE_CODE:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
stage="${1:?}";family="${2:-}"
if [[ "$stage" == tabular ]]; then
    "${MPOWER_BENCHMARK_PYTHON:?}" -u scripts/run_tabular_literature.py
    exit
fi
if [[ "$family" == hear ]]; then
    methods=(hear_lp_frozen hear_lp_full hear_lp_lora)
else
    methods=(hubert_ce hubert_detached_supcon)
fi
if [[ "$stage" == smoke ]]; then
    for method in "${methods[@]}";do
        "${MPOWER_BENCHMARK_PYTHON:?}" -u scripts/run_literature_benchmarks.py --method "$method" --smoke
    done
    "$MPOWER_BENCHMARK_PYTHON" -u scripts/audit_literature_smoke.py --family "$family"
else
    method="${methods[${SLURM_ARRAY_TASK_ID:?}]}"
    "${MPOWER_BENCHMARK_PYTHON:?}" -u scripts/run_literature_benchmarks.py --method "$method"
fi
