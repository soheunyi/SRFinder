#!/bin/bash
#SBATCH --job-name=affine-full
#SBATCH --partition=all
#SBATCH --array=0-119%120
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=data/refit_bootstrap/continuous_affine_full_v1/slurm-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/export/soheuny/SRFinder/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python \
    run_files/run_continuous_affine_full.py \
    --shard-index "$SLURM_ARRAY_TASK_ID" --n-shards 120
