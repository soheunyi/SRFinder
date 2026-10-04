#!/bin/bash
# Build the C++ bootstrap for the affine KS test and run its validation gate.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
g++ -O2 -std=c++17 -fPIC -shared -fno-fast-math -ffp-contract=off \
    run_files/affine_envelope_kernel.cpp -o run_files/affine_envelope_kernel.so
g++ -O2 -std=c++17 -fPIC -shared -fopenmp -fno-fast-math -ffp-contract=off \
    run_files/affine_ks_bootstrap_kernel.cpp -o run_files/affine_ks_bootstrap_kernel.so
python_bin=${PYTHON_BIN:-/home/export/soheuny/.conda/envs/coffea_torch/bin/python}
"$python_bin" run_files/test_affine_ks_fast.py "$@"
