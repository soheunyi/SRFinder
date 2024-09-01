#!/bin/bash
# specify partition to run in (statds or phil)
#SBATCH --partition statds
#SBATCH --ntasks 1

# number of CPUs and amount of memory you need:
#SBATCH --cpus-per-task 16
#SBATCH --mem-per-cpu 16G

# number of GPUs you need
#SBATCH --gres=gpu:1

# maximum time limit for task (up to 48 hours, 48:00:00)
#SBATCH --time=12:00:00

# email you when job is finished:
#SBATCH --mail-user=soheuny@andrew.cmu.edu
#SBATCH --mail-type=END

## Choose one:

CONFIG_FILENAME="counting_test_more_CR_train_data.yml"
RUN_FILENAME="run_counting_test_v1.py"
CONDA_ENV_NAME="coffea_torch"

PYTHON="/home/export/soheuny/.conda/envs/$CONDA_ENV_NAME/bin/python"

export TQDM_MININTERVAL=5
# to run a Python script in a Conda environment called ENV
cd ..
srun $PYTHON $RUN_FILENAME --config configs/$CONFIG_FILENAME
