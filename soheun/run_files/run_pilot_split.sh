#!/bin/bash
#SBATCH --job-name=pilot-split
#SBATCH --partition=all
#SBATCH --array=0-29%30
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=data/refit_bootstrap/pilot_split/slurm-%A_%a.out
set -euo pipefail
cd /home/export/soheuny/SRFinder/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python run_files/pilot_split.py   --shard-index "${SLURM_ARRAY_TASK_ID}" --n-shards 30
