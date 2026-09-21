#!/bin/bash
#SBATCH --job-name=calib-variants
#SBATCH --partition=all
#SBATCH --array=0-49%50
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=data/refit_bootstrap/calib_variants_eta1/slurm-%A_%a.out
set -euo pipefail
cd /home/export/soheuny/SRFinder/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python run_files/calibration_variants.py   --shard-index "${SLURM_ARRAY_TASK_ID}" --n-shards 50
