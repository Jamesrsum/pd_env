#!/usr/bin/env bash
# Keep the Conda Python/SQLite ICU runtime and C++ library compatible with the
# PyPI CUDA wheel. This changes only this child process, never the base env.
set -euo pipefail
xgb_python="${MPOWER_BENCHMARK_PYTHON:-python}"
xgb_cpp_runtime="$("$xgb_python" -c 'import sys; from pathlib import Path; print(Path(sys.base_prefix)/"lib/libstdc++.so.6")')"
if [[ -f "$xgb_cpp_runtime" ]];then
    export LD_PRELOAD="$xgb_cpp_runtime${LD_PRELOAD:+:$LD_PRELOAD}"
fi
exec "$xgb_python" "$@"
