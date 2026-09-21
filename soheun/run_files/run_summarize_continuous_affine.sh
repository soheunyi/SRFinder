#!/bin/bash
#SBATCH --job-name=summarize-continuous-affine
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=data/refit_bootstrap/continuous_affine_eta1_sr020_v1/summary-%j.out
set -euo pipefail
cd /home/export/soheuny/SRFinder/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python run_files/summarize_continuous_affine.py --require-complete
