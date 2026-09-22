#!/bin/bash
#SBATCH --job-name=step2-calib
#SBATCH --partition=all
#SBATCH --array=0-9%10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=/home/export/soheuny/SRFinder/soheun/data/refit_bootstrap/step2_calibration_audit_v1/slurm-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/export/soheuny/SRFinder/issue1-centered-poisson/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python \
    run_files/step2_calibration_audit.py --index "$SLURM_ARRAY_TASK_ID"
