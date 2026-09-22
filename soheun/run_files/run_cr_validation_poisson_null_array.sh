#!/bin/bash
#SBATCH --job-name=cr-null
#SBATCH --partition=all
#SBATCH --array=0-79%80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --requeue
#SBATCH --output=/home/export/soheuny/SRFinder/soheun/data/refit_bootstrap/cr_validation_poisson_null_v1/slurm-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/export/soheuny/SRFinder/issue1-centered-poisson/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python \
    run_files/cr_validation_poisson_null.py --index "$SLURM_ARRAY_TASK_ID"
