#!/bin/bash
#SBATCH --job-name=aggregate-draft-power
#SBATCH --partition=all
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=data/refit_bootstrap/draft_eta2_logitcap10_v1/aggregate-slurm-%j.out

set -euo pipefail
cd /home/export/soheuny/SRFinder/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python run_files/aggregate_draft_power.py
