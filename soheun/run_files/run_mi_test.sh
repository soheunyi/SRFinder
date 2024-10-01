#!/bin/bash
# specify partition to run in (statds or phil)
#SBATCH --partition statds
#SBATCH --ntasks 1

# number of CPUs and amount of memory you need:
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 8G

# number of GPUs you need
#SBATCH --gres=gpu:1

# maximum time limit for task (up to 48 hours, 48:00:00)
#SBATCH --time=24:00:00

# email you when job is finished:
#SBATCH --mail-user=no.reply@gmail.com
#SBATCH --mail-type=END

## Choose one:

RUN_FILENAME="mi_test.py"
CONDA_ENV_NAME="coffea_torch"

PYTHON="/home/export/soheuny/.conda/envs/$CONDA_ENV_NAME/bin/python"

SEED_START=00
SEED_END=10

# to run a Python script in a Conda environment called ENV
cd ..
srun $PYTHON $RUN_FILENAME --seed-start $SEED_START --seed-end $SEED_END
