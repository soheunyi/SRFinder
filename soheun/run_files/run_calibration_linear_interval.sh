#!/bin/bash
#SBATCH --job-name=linear-interval-calibration
#SBATCH --partition=all
#SBATCH --array=0-19%20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=data/refit_bootstrap/linear_interval_eta1_sr020_v1/slurm-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/export/soheuny/SRFinder/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python run_files/calibration_linear_interval.py --shard "$SLURM_ARRAY_TASK_ID" --shards 20
