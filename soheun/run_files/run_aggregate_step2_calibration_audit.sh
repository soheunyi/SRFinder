#!/bin/bash
#SBATCH --job-name=step2-calib-aggregate
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/export/soheuny/SRFinder/soheun/data/refit_bootstrap/step2_calibration_audit_v1/aggregate-slurm-%j.out
set -euo pipefail
cd /home/export/soheuny/SRFinder/issue1-centered-poisson/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python \
    run_files/step2_calibration_audit.py --aggregate
