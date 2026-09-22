#!/bin/bash
#SBATCH --job-name=phase5_bench
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.out

# Phase 5: benchmark K in {50, 100, 200}, one array task per K so they run
# concurrently on separate GPUs.
#
#   sbatch phase5/submit_benchmark.sbatch
#
# Memory: 32G. The (N, K, 16) stacked tensors are ~12 GB at K=200, but the cost
# is ~2x and SUSTAINED, not transient: StackedFvTClassifier.fit keeps
# self.train_datasets (the per-member list) alive for the whole run alongside
# the datamodule's stacked copy, so both exist at once. K=200 may therefore not
# fit in 32G. If it does not, the failure is cheap and early -- the peak is
# reached while the datasets are being built, in the first few minutes -- and
# that one task can be resubmitted with a larger --mem. The run records peak
# RSS per estimator so the projection is measured rather than assumed.
#
# Override per submission:  sbatch --mem=48G phase5/submit_benchmark.sbatch
#
# To add K=300 later (only if K=200 is still clearly improving):
#   KS=300 sbatch --array=0-0 phase5/submit_benchmark.sbatch

set -euo pipefail

SRFINDER=/home/export/soheuny/SRFinder/soheun
PYTHON=/home/export/soheuny/.conda/envs/coffea_torch/bin/python

KS=${KS:-"50 100 200"}
MAX_EPOCHS=${MAX_EPOCHS:-20}
read -ra K_ARR <<< "$KS"
K=${K_ARR[${SLURM_ARRAY_TASK_ID:-0}]}

cd "$SRFINDER"

echo "=== phase 5 benchmark ==="
echo "node : $(hostname)"
echo "K    : $K"
echo "mem  : ${SLURM_MEM_PER_NODE:-?} MB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$PYTHON" phase5/benchmark.py --num-stacks "$K" --max-epochs "$MAX_EPOCHS"
