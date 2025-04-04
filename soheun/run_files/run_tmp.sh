#!/bin/bash
# specify partition to run in (statds or phil)
#SBATCH --partition phil_condo
#SBATCH --ntasks 1

# number of CPUs and amount of memory you need:
#SBATCH --cpus-per-task 16
#SBATCH --mem-per-cpu 8G

# number of GPUs you need
#SBATCH --gres=gpu:1

# maximum time limit for task (up to 48 hours, 48:00:00)
#SBATCH --time=36:00:00

# email you when job is finished:
#SBATCH --mail-user=soheuny@andrew.cmu.edu
#SBATCH --mail-type=END

## Choose one:

RUN_FILENAME="save_corrected_pulls_new.py"
CONDA_ENV_NAME="coffea_torch"
# ARGS="--ds-start 47 --ds-end 50"

PYTHON="/home/export/soheuny/.conda/envs/$CONDA_ENV_NAME/bin/python"

# to run a Python script in a Conda environment called ENV
cd ..
# srun $PYTHON $RUN_FILENAME $ARGS
srun $PYTHON $RUN_FILENAME
