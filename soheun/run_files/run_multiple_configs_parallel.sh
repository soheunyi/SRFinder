#!/bin/bash
#SBATCH --partition statds 
#SBATCH --ntasks 1

# number of CPUs and amount of memory you need:
#SBATCH --cpus-per-task 16
#SBATCH --mem-per-cpu 8G

# number of GPUs you need
#SBATCH --gres=gpu:1

# maximum time limit for task (up to 48 hours, 48:00:00)
#SBATCH --time=24:00:00

# email you when job is finished:
#SBATCH --mail-user=no.reply@gmail.com
#SBATCH --mail-type=END

## Choose one:

CONDA_ENV_NAME="coffea_torch"
PYTHON="/home/export/soheuny/.conda/envs/$CONDA_ENV_NAME/bin/python"

# to run a Python script in a Conda environment called ENV
cd ..
CONFIGS_LIST="" # Should be modified by the user with something like "-c config1.yml -c config2.yml"
NPROCS=4
echo $CONFIGS_LIST
srun $PYTHON run_multiple_configs_parallel.py $CONFIGS_LIST -p $NPROCS
