#!/bin/bash
#SBATCH --job-name=affine-aggregate
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=data/refit_bootstrap/continuous_affine_full_v1/aggregate-slurm-%j.out
set -euo pipefail
cd /home/export/soheuny/SRFinder/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python \
    run_files/aggregate_continuous_affine_full.py
