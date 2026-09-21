#!/bin/bash
#SBATCH --job-name=fig-owl
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=data/refit_bootstrap/draft_eta2_logitcap10_v1/figure_logs/owl-%j.out

set -euo pipefail
cd /home/export/soheuny/SRFinder/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python run_files/figure_scripts/on_which_to_learn.py
