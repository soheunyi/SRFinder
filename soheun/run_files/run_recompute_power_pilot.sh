#!/bin/bash
#SBATCH --job-name=refit-power-pilot
#SBATCH --partition=all
#SBATCH --array=0-15%16
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=data/refit_bootstrap/power_pilot_eta2_logitcap10_v1/slurm-%A_%a.out

set -euo pipefail

cd /home/export/soheuny/SRFinder/soheun

/home/export/soheuny/.conda/envs/coffea_torch/bin/python \
  run_files/recompute_power_pilot.py \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --n-shards 16 \
  --workers 4
