#!/bin/bash
#SBATCH --job-name=phase1_baseline
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Phase 1 reference run: train the same dev stack twice on one GPU and check
# that the two fingerprints agree.  Both runs land on the same node and GPU,
# which isolates run-to-run nondeterminism from hardware differences.
#
#   sbatch phase1/submit_baseline.sbatch
#   NUM_STACKS=2 MAX_EPOCHS=3 OUT_ROOT=phase1_runs/smoke \
#       sbatch phase1/submit_baseline.sbatch

set -euo pipefail

SRFINDER=/home/export/soheuny/SRFinder/soheun
PYTHON=/home/export/soheuny/.conda/envs/coffea_torch/bin/python

NUM_STACKS=${NUM_STACKS:-5}
MAX_EPOCHS=${MAX_EPOCHS:-20}
OUT_ROOT=${OUT_ROOT:-phase1_runs/dev}
PROBE_SIZE=${PROBE_SIZE:-20000}

cd "$SRFINDER"

echo "=== phase 1 baseline ==="
echo "node        : $(hostname)"
echo "num_stacks  : $NUM_STACKS"
echo "max_epochs  : $MAX_EPOCHS"
echo "out_root    : $OUT_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for TAG in A B; do
    echo "=== run $TAG ==="
    srun "$PYTHON" phase1/run_baseline.py \
        --tag "$TAG" \
        --out-root "$OUT_ROOT" \
        --num-stacks "$NUM_STACKS" \
        --max-epochs "$MAX_EPOCHS" \
        --probe-size "$PROBE_SIZE"
done

echo "=== comparison ==="
"$PYTHON" phase1/compare.py "$OUT_ROOT/A" "$OUT_ROOT/B" | tee "$OUT_ROOT/phase1_report.txt"
