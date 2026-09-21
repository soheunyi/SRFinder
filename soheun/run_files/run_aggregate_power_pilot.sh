#!/bin/bash
#SBATCH --job-name=aggregate-power-pilot
#SBATCH --partition=all
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=data/refit_bootstrap/power_pilot_eta2_logitcap10_v1/aggregate-slurm-%j.out

set -euo pipefail
cd /home/export/soheuny/SRFinder/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python run_files/aggregate_power_pilot.py
