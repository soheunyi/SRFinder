from copy import deepcopy
import datetime
import hashlib
import json
import os
import stat
import time
import numpy as np
import sys
import logging
import pandas as pd
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

N_STATDS = 2
N_PHIL = 0
N_RUNFILES = 6


# STEP = 1
# GROUP_KEYS = [
#      "dataset.signal_ratio",
#      "base_fvt.model_seed",
#      "base_fvt.train_seed",
#      "base_fvt.data_seed",
#  ]
# EXPERIMENT_NAME = "base_fvt_training_ensemble_ZH4b"
# BASE_CONFIG_FILENAME = "better_fvt_training.yml"

# STEP = 2
# GROUP_KEYS = [
#     "dataset.signal_ratio",
#     "smeared_fvt.model_seed",
#     "smeared_fvt.train_seed",
#     "smeared_fvt.data_seed",
#     "smearing.noise_scale",
# ]
# EXPERIMENT_NAME = "smeared_fvt_training_ensemble"
# BASE_CONFIG_FILENAME = "smeared_fvt_training.yml"

STEP = 3
GROUP_KEYS = [
     "signal_region.4b_in_SR",
     "signal_region.4b_in_CR",
     "smearing.noise_scale",
     "dataset.signal_ratio",
     "CR_fvt.train_seed",
     "CR_fvt.model_seed",
     "CR_fvt.data_seed",
]
EXPERIMENT_NAME = "CR_fvt_training_ensemble_max"
BASE_CONFIG_FILENAME = "CR_fvt_training_original_features.yml"

config_filenames, configs_to_run = write_and_get_configs_to_run(
    STEP, EXPERIMENT_NAME, BASE_CONFIG_FILENAME
)


def get_values_to_group(config: dict[str, any], group_keys: list[str]):
    result = {}
    for key in group_keys:
        config_iter = deepcopy(config)
        gkeys = key.split(".")
        for gkey in gkeys:
            try:
                config_iter = config_iter[gkey]
            except KeyError:
                raise KeyError(f"Key {key} not found in config {config}")
        result[key] = config_iter
    return result


def group_configs(configs: list[dict[str, any]], group_keys: list[str]):
    configs_to_group = pd.DataFrame(
        [get_values_to_group(config, group_keys) for config in configs],
        index=config_filenames,
    )
    gpby = configs_to_group.groupby(group_keys)
    grouped_config_filenames = []
    for gkeys in gpby.groups:
        grouped_config_filenames.append(gpby.get_group(gkeys).index.tolist())
    return grouped_config_filenames


grouped_config_filenames = group_configs(configs_to_run, group_keys=GROUP_KEYS)
n_groups = len(grouped_config_filenames)
groups_alloc = {i: {"groups": [], "config_filenames": []} for i in range(N_RUNFILES)}
for i in range(n_groups):
    runfile_idx = i % N_RUNFILES
    group = grouped_config_filenames[i]
    groups_alloc[runfile_idx]["groups"].extend([i] * len(group))
    groups_alloc[runfile_idx]["config_filenames"].extend(group)


RUN_PARTITIONS = (
    ["statds"] * N_STATDS
    + ["phil_condo"] * N_PHIL
    + ["cmist_condo"] * (N_RUNFILES - N_STATDS - N_PHIL)
)

logging.info(f"Partitions: {RUN_PARTITIONS}")
# Get input
input("Press Enter to continue...")


for k, v in groups_alloc.items():
    config_filenames = v["config_filenames"]
    config_filenames = [f"configs/tmp/{cfg}" for cfg in config_filenames]
    groups = v["groups"]
    if len(config_filenames) == 0:
        continue

    # get timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # create hash from timestamp
    hash = hashlib.sha256(timestamp.encode()).hexdigest()[:8]
    RUN_PARTITION = RUN_PARTITIONS[k]
    RUN_FILENAME = f"run_{EXPERIMENT_NAME}_{k}_{hash}.sh"
    RUN_ARGS_FILENAME = f"run_{EXPERIMENT_NAME}_{k}_{hash}_args.json"

    with open(f"args/{RUN_ARGS_FILENAME}", "w") as f:
        json.dump({"config_filenames": config_filenames, "groups": groups}, f)

    with open("run_stacked_configs.sh", "r") as f:
        run_script = f.read()

    run_script = run_script.replace(
        "#SBATCH --partition statds",
        f"#SBATCH --partition {RUN_PARTITION}",
    )
    run_script = run_script.replace(
        'ARGS_FILENAME=""', f'ARGS_FILENAME="run_files/args/{RUN_ARGS_FILENAME}"'
    )
    with open(RUN_FILENAME, "w") as f:
        f.write(run_script)

    # Make run script executable and run it
    st = os.stat(RUN_FILENAME)
    os.chmod(RUN_FILENAME, st.st_mode | stat.S_IEXEC)
    os.system(f"sbatch {RUN_FILENAME}")
    time.sleep(1)
    os.remove(RUN_FILENAME)
