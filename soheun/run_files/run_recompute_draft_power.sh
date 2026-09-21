#!/bin/bash
#SBATCH --job-name=refit-draft-power
#SBATCH --partition=all
#SBATCH --array=0-63%64
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=data/refit_bootstrap/draft_eta2_logitcap10_v1/slurm-%A_%a.out

set -euo pipefail

cd /home/export/soheuny/SRFinder/soheun

/home/export/soheuny/.conda/envs/coffea_torch/bin/python \
  run_files/recompute_draft_power.py \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --n-shards 64 \
  --workers 4
