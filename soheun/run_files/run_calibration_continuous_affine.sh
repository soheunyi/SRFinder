#!/bin/bash
#SBATCH --job-name=continuous-affine-ks
#SBATCH --partition=all
#SBATCH --array=0-49%50
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=data/refit_bootstrap/continuous_affine_eta1_sr020_v1/slurm-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/export/soheuny/SRFinder/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python run_files/calibration_continuous_affine.py --shard "$SLURM_ARRAY_TASK_ID" --shards 50
