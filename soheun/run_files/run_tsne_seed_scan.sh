#!/bin/bash
#SBATCH --job-name=tsne-seed-scan
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=data/tsne_seed_scan/scan-slurm-%j.out

set -euo pipefail
cd /home/export/soheuny/SRFinder/soheun
mkdir -p data/tsne_seed_scan
/home/export/soheuny/.conda/envs/coffea_torch/bin/python run_files/tsne_seed_scan.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 --n-points 20000 --tag scan
