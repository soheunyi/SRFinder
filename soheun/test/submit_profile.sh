#!/usr/bin/env bash
#SBATCH --partition statds
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mail-user=soheuny@andrew.cmu.edu
#SBATCH --mail-type=END

CONDA_ENV_NAME="coffea_torch"
PYTHON="/home/export/soheuny/.conda/envs/$CONDA_ENV_NAME/bin/python"

echo "Starting FvTClassifier profiling"
echo "Using CUDA: $CUDA_VISIBLE_DEVICES"

# Set Python path to include the project root
export PYTHONPATH="/home/export/soheuny/SRFinder/soheun:$PYTHONPATH"

cd /home/export/soheuny/SRFinder/soheun
srun $PYTHON test/profile_fit.py 