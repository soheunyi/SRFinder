#!/bin/bash
#SBATCH --job-name=calib-seeds
#SBATCH --partition=all
#SBATCH --array=0-19%20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=/home/export/soheuny/SRFinder/soheun/data/refit_bootstrap/calibration_seed_sweep_v1/slurm-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/export/soheuny/SRFinder/issue1-centered-poisson/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python \
    run_files/calibration_seed_sweep.py --index "$SLURM_ARRAY_TASK_ID"
