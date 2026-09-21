#!/bin/bash
#SBATCH --job-name=affine-figures
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=data/refit_bootstrap/continuous_affine_full_v1/figures-slurm-%j.out
set -euo pipefail
cd /home/export/soheuny/SRFinder/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python \
    run_files/generate_continuous_affine_power_figures.py
