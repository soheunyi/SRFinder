#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

g++ -O3 -std=c++17 -fPIC -shared -fno-fast-math -ffp-contract=off \
    run_files/poisson_multiplier_ks_kernel.cpp \
    -o run_files/poisson_multiplier_ks_kernel.so

python_bin=${PYTHON_BIN:-/home/export/soheuny/.conda/envs/coffea_torch/bin/python}
"$python_bin" run_files/test_poisson_multiplier_ks.py
"$python_bin" run_files/test_poisson_multiplier_ks_compiled.py
"$python_bin" run_files/test_affine_poisson_multiplier_ks.py
