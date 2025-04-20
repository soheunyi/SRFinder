import os
import stat
import time
import numpy as np
import sys
import logging
from get_configs_to_run import write_and_get_configs_to_run

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

sys.path.append("..")
from training_info import TrainingInfo

# update metadata before running
TrainingInfo.update_metadata()


######################################################################################
######################### Set Experiment Name and N Runfiles #########################
######################################################################################

PROB_STATDS = 0.4
PROB_PHIL = 0.4
N_RUNFILES = 10
NPROCS = 10

# STEP = 1
# EXPERIMENT_NAME = "base_fvt_training_ensemble"
# BASE_CONFIG_FILENAME = "better_fvt_training.yml"

# Was running this at March 25
# STEP = 1
# EXPERIMENT_NAME = "base_fvt_training_ensemble_HH4b_800"
# BASE_CONFIG_FILENAME = "better_fvt_training.yml"

# STEP = 2
# EXPERIMENT_NAME = "smeared_fvt_training_ensemble"
# BASE_CONFIG_FILENAME = "smeared_fvt_training.yml"

# STEP = 2
# EXPERIMENT_NAME = "smeared_fvt_training_ensemble_HH4b_400"
# BASE_CONFIG_FILENAME = "smeared_fvt_training.yml"

# STEP = 2
# EXPERIMENT_NAME = "smeared_fvt_training_ensemble_HH4b_800"
# BASE_CONFIG_FILENAME = "smeared_fvt_training.yml"

# STEP = 3
# EXPERIMENT_NAME = "CR_fvt_training_ensemble_max"
# BASE_CONFIG_FILENAME = "CR_fvt_training_original_features.yml"

STEP = 3
EXPERIMENT_NAME = "CR_fvt_training_ensemble_mean"
BASE_CONFIG_FILENAME = "CR_fvt_training_original_features.yml"

# STEP = 3
# EXPERIMENT_NAME = "CR_fvt_training_ensemble_max_HH4b_400"
# BASE_CONFIG_FILENAME = "CR_fvt_training_original_features.yml"

# STEP = 3
# EXPERIMENT_NAME = "CR_fvt_training_ensemble_max_HH4b_800"
# BASE_CONFIG_FILENAME = "CR_fvt_training_original_features.yml"

# STEP = 4
# EXPERIMENT_NAME = "mi_test"
# BASE_CONFIG_FILENAME = "mi_test.yml"

CONFIG_STRINGS = write_and_get_configs_to_run(
    STEP, EXPERIMENT_NAME, BASE_CONFIG_FILENAME
)
CONFIG_CHUNKS = {i: [] for i in range(N_RUNFILES)}
for i, config_string in enumerate(CONFIG_STRINGS):
    CONFIG_CHUNKS[i % N_RUNFILES].append(config_string)

RUN_PARTITIONS = []
for _ in range(N_RUNFILES):
    if np.random.rand() < PROB_STATDS:
        RUN_PARTITIONS.append("statds")
    elif np.random.rand() < PROB_PHIL + PROB_STATDS:
        RUN_PARTITIONS.append("phil_condo")
    else:
        RUN_PARTITIONS.append("cmist_condo")

logging.info(
    f"n_tasks (runfiles) {len(CONFIG_CHUNKS)}, max n_tasks per runfile {max([len(chunk) for chunk in CONFIG_CHUNKS.values()])}"
)
logging.info(f"Partitions: {RUN_PARTITIONS}")
# Get input
input("Press Enter to continue...")


for k, config_strings in CONFIG_CHUNKS.items():
    if len(config_strings) == 0:
        continue

    RUN_PARTITION = RUN_PARTITIONS[k]
    RUN_FILENAME = f"run_{EXPERIMENT_NAME}_{k}.sh"

    with open("run_multiple_configs_parallel.sh", "r") as f:
        run_script = f.read()

    run_script = run_script.replace(
        'CONFIGS_LIST=""', f'CONFIGS_LIST="{" ".join(config_strings)}"'
    )
    run_script = run_script.replace("NPROCS=4", f"NPROCS={NPROCS}")
    run_script = run_script.replace(
        "#SBATCH --partition statds",
        f"#SBATCH --partition {RUN_PARTITION}",
    )

    with open(RUN_FILENAME, "w") as f:
        f.write(run_script)

    # Make run script executable and run it
    st = os.stat(RUN_FILENAME)
    os.chmod(RUN_FILENAME, st.st_mode | stat.S_IEXEC)
    os.system(f"sbatch {RUN_FILENAME}")
    time.sleep(0.5)
    os.remove(RUN_FILENAME)
