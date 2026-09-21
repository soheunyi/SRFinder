#!/bin/bash
#SBATCH --job-name=calib-seeds-aggregate
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/export/soheuny/SRFinder/soheun/data/refit_bootstrap/calibration_seed_sweep_v1/aggregate-slurm-%j.out
set -euo pipefail
cd /home/export/soheuny/SRFinder/issue1-centered-poisson/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python \
    run_files/calibration_seed_sweep.py --aggregate
