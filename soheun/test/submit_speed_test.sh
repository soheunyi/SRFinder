#!/usr/bin/env bash
#SBATCH --partition statds
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-user=your.email@domain.com
#SBATCH --mail-type=END

# Load modules or activate environment as needed:
# module load anaconda
# source activate coffea_torch

echo "Running FvTClassifier speed test"
cd "$SLURM_SUBMIT_DIR"
python soheun/test/test_training_speed.py