#!/bin/bash
#SBATCH --job-name=tsne-rescan
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=data/tsne_seed_scan/rescan-slurm-%j.out

set -euo pipefail
cd /home/export/soheuny/SRFinder/soheun
/home/export/soheuny/.conda/envs/coffea_torch/bin/python run_files/tsne_seed_scan.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 --n-points 20000 --tag wscan
