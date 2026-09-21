#!/bin/bash
#SBATCH --job-name=refit-null-bootstrap
#SBATCH --partition=statds
#SBATCH --array=0-4%5
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=data/refit_bootstrap/null_eta2_logitcap10_v1/slurm-%A_%a.out

set -euo pipefail

cd /home/export/soheuny/SRFinder/soheun

/home/export/soheuny/.conda/envs/coffea_torch/bin/python \
  run_files/recompute_null_rejection_rates.py \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --n-shards 5 \
  --workers 4
