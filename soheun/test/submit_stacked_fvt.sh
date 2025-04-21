#!/usr/bin/env bash
#SBATCH --partition statds
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mail-user=soheuny@andrew.cmu.edu
#SBATCH --mail-type=END

CONDA_ENV_NAME="coffea_torch"
PYTHON="/home/export/soheuny/.conda/envs/$CONDA_ENV_NAME/bin/python"

echo "Running FvTClassifier stacked training"
cd ..
srun $PYTHON test/test_stacked_fvt.py