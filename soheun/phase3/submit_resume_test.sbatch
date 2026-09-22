#!/bin/bash
#SBATCH --job-name=phase3_resume
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=%x-%j.out

# Phase 3 resume gate. Trains one group straight through, then trains the same
# group again with an interruption after a prescribed epoch, and compares.
#
# --requeue and --signal=B:USR1@120 are the preemption path from issue #2.
# Lightning's SLURMEnvironment already installs the SIGUSR1 handler on its own
# ("SLURM auto-requeueing enabled" appears in the logs), and with a
# ModelCheckpoint registered it now has something to write.
#
#   sbatch phase3/submit_resume_test.sbatch

set -euo pipefail

SRFINDER=/home/export/soheuny/SRFinder/soheun
PYTHON=/home/export/soheuny/.conda/envs/coffea_torch/bin/python

CAMPAIGN=${CAMPAIGN:-phase3_resume}
NUM_STACKS=${NUM_STACKS:-5}
MAX_EPOCHS=${MAX_EPOCHS:-20}
STOP_AFTER=${STOP_AFTER:-7}

cd "$SRFINDER"

echo "=== phase 3 resume gate ==="
echo "node        : $(hostname)"
echo "campaign    : $CAMPAIGN"
echo "num_stacks  : $NUM_STACKS"
echo "max_epochs  : $MAX_EPOCHS"
echo "stop_after  : epoch $STOP_AFTER"
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== uninterrupted reference ==="
srun "$PYTHON" phase3/run_resumable.py \
    --campaign-id "$CAMPAIGN" --group-id whole \
    --num-stacks "$NUM_STACKS" --max-epochs "$MAX_EPOCHS"

echo "=== segment 1: stop after epoch $STOP_AFTER ==="
srun "$PYTHON" phase3/run_resumable.py \
    --campaign-id "$CAMPAIGN" --group-id split \
    --num-stacks "$NUM_STACKS" --max-epochs "$MAX_EPOCHS" \
    --stop-after-epoch "$STOP_AFTER"

echo "=== segment 2: resume ==="
srun "$PYTHON" phase3/run_resumable.py \
    --campaign-id "$CAMPAIGN" --group-id split \
    --num-stacks "$NUM_STACKS" --max-epochs "$MAX_EPOCHS" \
    --resume

echo "=== comparison ==="
"$PYTHON" phase3/compare_resume.py \
    "training_runs/$CAMPAIGN/whole" "training_runs/$CAMPAIGN/split" \
    | tee "training_runs/$CAMPAIGN/phase3_report.txt"
