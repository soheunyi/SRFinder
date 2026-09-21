#!/bin/bash
#SBATCH --job-name=calib-split025
#SBATCH --partition=all
#SBATCH --array=0-63%64
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=data/refit_bootstrap/calib_split025_all/slurm-%A_%a.out
set -euo pipefail
cd /home/export/soheuny/SRFinder/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python run_files/calibration_split_all.py   --shard-index "${SLURM_ARRAY_TASK_ID}" --n-shards 64
