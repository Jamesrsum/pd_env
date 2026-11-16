#!/usr/bin/env bash
#SBATCH --job-name=mpower_adaptation_setup
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077
cd "${MPOWER_REPO_ROOT:?}"
: "${MPOWER_ADAPT_ROOT:?}" "${MPOWER_MODEL_ROOT:?}" "${MPOWER_BENCHMARK_PYTHON:?}"
mkdir -p "$MPOWER_ADAPT_ROOT/dependencies"
"$MPOWER_BENCHMARK_PYTHON" -m pip install --no-cache-dir --no-deps --target "$MPOWER_ADAPT_ROOT/dependencies" torchvision==0.29.0
export PYTHONPATH="$MPOWER_ADAPT_ROOT/dependencies:$PWD"
"$MPOWER_BENCHMARK_PYTHON" -u scripts/prepare_adaptation_assets.py
