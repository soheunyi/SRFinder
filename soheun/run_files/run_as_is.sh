#!/bin/bash
#SBATCH --partition statds 
#SBATCH --ntasks 1

# number of CPUs and amount of memory you need:
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 8G

# number of GPUs you need
#SBATCH --gres=gpu:1

# maximum time limit for task (up to 48 hours, 48:00:00)
#SBATCH --time=48:00:00

# email you when job is finished:
#SBATCH --mail-user=no.reply@gmail.com
#SBATCH --mail-type=END

## Choose one:

CONDA_ENV_NAME="coffea_torch"
PYTHON="/home/export/soheuny/.conda/envs/$CONDA_ENV_NAME/bin/python"

CONFIG_FILENAME="" # Should be modified by the user

# to run a Python script in a Conda environment called ENV
cd ..
srun $PYTHON run_with_config.py --config configs/$CONFIG_FILENAME
