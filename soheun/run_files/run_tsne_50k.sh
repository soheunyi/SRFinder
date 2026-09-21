#!/bin/bash
#SBATCH --job-name=tsne-50k
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=data/tsne_seed_scan/tsne50k-slurm-%j.out

set -euo pipefail
cd /home/export/soheuny/SRFinder/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python run_files/tsne_seed_scan.py \
    --seeds 2 7 9 --n-points 50000 --tag k50
