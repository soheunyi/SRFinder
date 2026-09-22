#!/bin/bash
#SBATCH --job-name=phase2_permutation
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Phase 2 end-to-end gate: train the same five estimators twice, in opposite
# stack orders, with initialization derived from identity. Reordering must
# change neither initial parameters nor predictions.
#
#   sbatch phase2/submit_permutation.sbatch

set -euo pipefail

SRFINDER=/home/export/soheuny/SRFinder/soheun
PYTHON=/home/export/soheuny/.conda/envs/coffea_torch/bin/python

NUM_STACKS=${NUM_STACKS:-5}
MAX_EPOCHS=${MAX_EPOCHS:-20}
OUT_ROOT=${OUT_ROOT:-phase2_runs/perm}

cd "$SRFINDER"

echo "=== phase 2 permutation gate ==="
echo "node       : $(hostname)"
echo "num_stacks : $NUM_STACKS"
echo "max_epochs : $MAX_EPOCHS"
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== unit gate ==="
"$PYTHON" phase2/test_identity.py

for spec in "fwd forward" "rev reverse"; do
    set -- $spec
    echo "=== run $1 ($2) ==="
    srun "$PYTHON" phase2/run_with_identities.py \
        --tag "$1" --order "$2" \
        --out-root "$OUT_ROOT" \
        --num-stacks "$NUM_STACKS" \
        --max-epochs "$MAX_EPOCHS"
done

echo "=== comparison ==="
"$PYTHON" phase2/compare_permutation.py "$OUT_ROOT/fwd" "$OUT_ROOT/rev" \
    | tee "$OUT_ROOT/phase2_report.txt"
