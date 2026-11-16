#!/usr/bin/env bash
#SBATCH --job-name=mpower_adaptation_summary
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${MPOWER_CODE_ROOT:?}"
"${MPOWER_BENCHMARK_PYTHON:?}" scripts/summarize_adaptation.py
