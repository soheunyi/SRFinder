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

# # To run Step 1:
# CONFIG_FILENAME="better_fvt_training.yml"
# CONFIG_FILENAME="better_fvt_training_small.yml"
# RUN_FILENAME="run_step_1_base_fvt_training.py"

# # To run Step 2:
CONFIG_FILENAME="smeared_fvt_training_small.yml"
RUN_FILENAME="run_step_2_smeared_fvt_training.py"

# # To run Step 3:
# CONFIG_FILENAME="CR_fvt_training_repr.yml"
# CONFIG_FILENAME="CR_fvt_training_original_features.yml"
# CONFIG_FILENAME="CR_fvt_training_schedulefree.yml"
# RUN_FILENAME="run_step_3_define_CR_and_train_fvt.py"

PYTHON="/home/export/soheuny/.conda/envs/$CONDA_ENV_NAME/bin/python"

SEED_START=0
SEED_END=10
SIGNAL_RATIO=0.0

# to run a Python script in a Conda environment called ENV
cd ..
srun $PYTHON $RUN_FILENAME --config configs/$CONFIG_FILENAME --seed-start $SEED_START --seed-end $SEED_END --signal-ratio $SIGNAL_RATIO
