#!/bin/bash
#SBATCH --job-name=calib-split-eta2
#SBATCH --partition=all
#SBATCH --array=0-31%32
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=data/refit_bootstrap/calib_split025_all/eta2-slurm-%A_%a.out
set -euo pipefail
cd /home/export/soheuny/SRFinder/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python run_files/calibration_split_eta2.py   --shard-index "${SLURM_ARRAY_TASK_ID}" --n-shards 32
