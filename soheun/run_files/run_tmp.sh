#!/bin/bash
# specify partition to run in (statds or phil)
#SBATCH --partition phil_condo
#SBATCH --ntasks 1

# number of CPUs and amount of memory you need:
#SBATCH --cpus-per-task 16
#SBATCH --mem-per-cpu 8G

# number of GPUs you need
#SBATCH --gres=gpu:0

# maximum time limit for task (up to 48 hours, 48:00:00)
#SBATCH --time=36:00:00

# email you when job is finished:
#SBATCH --mail-user=soheunyi@gmail.com
#SBATCH --mail-type=END

## Choose one:

RUN_FILENAME="affine_correction_and_ks_bootstrap.py"
CONDA_ENV_NAME="coffea_torch"
ARGS="--experiment_name CR_fvt_training_ensemble_max --n_reps 1000 --signal_ratio 0.0 --cdf_mode mean"

PYTHON="/home/export/soheuny/.conda/envs/$CONDA_ENV_NAME/bin/python"

# to run a Python script in a Conda environment called ENV
cd ..
srun $PYTHON $RUN_FILENAME $ARGS
