#!/bin/bash
#SBATCH --job-name=regenerate-draft-figures
#SBATCH --partition=all
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=data/refit_bootstrap/draft_eta2_logitcap10_v1/figures-slurm-%j.out

set -euo pipefail

cd /home/export/soheuny/SRFinder/soheun

PYTHON=/home/export/soheuny/.conda/envs/coffea_torch/bin/python

"${PYTHON}" run_files/run_draft_figure_notebooks.py
"${PYTHON}" run_files/generate_continuous_affine_power_figures.py
