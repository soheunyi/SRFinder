#!/bin/bash
#SBATCH --job-name=tsne-members
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=data/tsne_seed_scan/members-slurm-%j.out

set -euo pipefail
cd /home/export/soheuny/SRFinder/soheun
PYTHON=/home/export/soheuny/.conda/envs/coffea_torch/bin/python
for m in 2 4; do
    "${PYTHON}" run_files/tsne_seed_scan.py \
        --seeds 2 9 --n-points 20000 --repr-members "$m" --tag "m${m}"
done
