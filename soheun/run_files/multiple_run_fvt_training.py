import yaml
import os
import stat
import time
from itertools import product
import numpy as np
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Packages loaded, except for custom modules")

sys.path.append("..")
from training_info import TrainingInfo
from utils import safe_dict

logging.info("Custom modules loaded")

# update metadata before running
TrainingInfo.update_metadata()


def find_and_check_hash_unique(hparams_filter: dict):
    """Finds a unique training info hash based on a filter.

    Args:
        hparams_filter: Dictionary containing filter criteria.

    Returns:
        The unique hash.

    Raises:
        AssertionError: If more or less than one hash is found.
    """
    hashes = TrainingInfo.find(hparams_filter)
    assert (
        len(hashes) == 1
    ), f"Expected 1 training info, found {len(hashes)} for filter: {hparams_filter}"
    return hashes[0]


def find_and_check_hashes_have_same_ms(hparams_filter: dict):
    hashes = TrainingInfo.find(hparams_filter)
    tinfo_0 = TrainingInfo.load(hashes[0])
    for hash in hashes:
        tinfo = TrainingInfo.load(hash)
        assert tinfo.ms_hash == tinfo_0.ms_hash
        assert np.all(tinfo.ms_idx == tinfo_0.ms_idx)
    return list(hashes)


def check_dataset_conditions(
    n_3b: int,
    ratio_4b: float,
    signal_ratio: float,
    signal_filename: str,
    seed: int,
):
    """Creates a lambda function to check dataset conditions."""
    return lambda dset_hparam: (
        dset_hparam["n_3b"] == n_3b
        and dset_hparam["ratio_4b"] == ratio_4b
        and dset_hparam["signal_ratio"] == signal_ratio
        and dset_hparam["signal_filename"] == signal_filename
        and dset_hparam["seed"] == seed
    )


######################################################################################
######################### Set Experiment Name and N Runfiles #########################
######################################################################################

PROB_STATDS = 1.0
N_RUNFILES = 5
NPROCS = 10

# EXPERIMENT_NAME = "better_base_fvt_training"
# BASE_CONFIG_FILENAME = "better_fvt_training.yml"

# EXPERIMENT_NAME = "base_fvt_training_ensemble"
# BASE_CONFIG_FILENAME = "better_fvt_training.yml"

# EXPERIMENT_NAME = "base_fvt_training_fixed_data_split"
# BASE_CONFIG_FILENAME = "better_fvt_training.yml"

# EXPERIMENT_NAME = "base_fvt_with_repr_norm"
# BASE_CONFIG_FILENAME = "base_fvt_with_repr_norm.yml"

# EXPERIMENT_NAME = "smeared_fvt_training"
# BASE_CONFIG_FILENAME = "smeared_fvt_training.yml"

# EXPERIMENT_NAME = "smeared_fvt_training_ensemble"
# BASE_CONFIG_FILENAME = "smeared_fvt_training.yml"

# EXPERIMENT_NAME = "smeared_fvt_training_repr_norm"
# BASE_CONFIG_FILENAME = "smeared_fvt_training.yml"

# EXPERIMENT_NAME = "CR_fvt_training_v2"
# BASE_CONFIG_FILENAME = "CR_fvt_training_original_features.yml"

# EXPERIMENT_NAME = "CR_fvt_training_repr_norm"
# BASE_CONFIG_FILENAME = "CR_fvt_training_original_features.yml"

# EXPERIMENT_NAME = "CR_fvt_training_ensemble_max_smeared"
# BASE_CONFIG_FILENAME = "CR_fvt_training_original_features.yml"

EXPERIMENT_NAME = "CR_fvt_training_ensemble_max_fvt"
BASE_CONFIG_FILENAME = "CR_fvt_training_original_features.yml"


######################################################################################
################################## Set What to Run ###################################
######################################################################################

# signal_ratios = [0.0]
# dataset_seeds = range(1)  # start with ten seeds, will be increased to fifty later
signal_ratios = [0.0, 0.005, 0.0075, 0.01, 0.02]
dataset_seeds = range(50)  # start with ten seeds, will be increased to fifty later
ensemble_seeds = range(1)

hparams_filter = {
    "experiment_name": EXPERIMENT_NAME,
    "dataset": lambda x: (
        safe_dict(x, "signal_ratio") in signal_ratios
        and safe_dict(x, "seed") in dataset_seeds
    ),
    "train_seed": lambda x: x in ensemble_seeds,
    "model_seed": lambda x: x in ensemble_seeds,
    "data_seed": lambda x: x in ensemble_seeds,
}
_, hparams = TrainingInfo.find(hparams_filter, return_hparams=True)
existing = {
    (
        hparam["dataset"]["signal_ratio"],
        hparam["dataset"]["seed"],
        hparam["train_seed"],
    )
    for hparam in hparams
}
targets = set(product(signal_ratios, dataset_seeds, ensemble_seeds))
targets = list(targets - existing)
# sort by seed and signal ratio
targets.sort(key=lambda x: (x[0], x[1], x[2]))
logging.info(f"Number of targets: {len(targets)}")


CONFIGS_TO_RUN = []
for signal_ratio, seed, ensemble_seed in targets:
    # postfix = f"{seed}_{noise_scale}_{signal_ratio}"
    postfix = f"{seed}_{signal_ratio}_{ensemble_seed}"

    NEW_CONFIG_FILENAME = f"{EXPERIMENT_NAME}_{postfix}.yml"

    # Update config file
    try:
        with open(f"../configs/{BASE_CONFIG_FILENAME}", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        logging.error(f"Base config file not found: {BASE_CONFIG_FILENAME}")
        raise e

    config["experiment_name"] = EXPERIMENT_NAME
    config["dataset"]["signal_ratio"] = signal_ratio
    config["dataset"]["seed"] = seed

    # Step 1 config
    # config["base_fvt"]["train_seed"] = ensemble_seed
    # config["base_fvt"]["model_seed"] = ensemble_seed
    # config["base_fvt"]["data_seed"] = ensemble_seed

    # Step 2 config
    # config["smeared_fvt"]["train_seed"] = ensemble_seed
    # config["smeared_fvt"]["model_seed"] = ensemble_seed
    # config["smeared_fvt"]["data_seed"] = ensemble_seed
    # config["smearing"]["noise_scale"] = 1.0
    # config["smearing"]["seed"] = seed
    # base_experiment_name = "base_fvt_training_ensemble"
    # config["base_experiment_name"] = base_experiment_name
    # config["base_experiment_hash"] = find_and_check_hash_unique(
    #     {
    #         "experiment_name": base_experiment_name,
    #         "aux_info_step": 1,
    #         "dataset": check_dataset_conditions(
    #             config["dataset"]["n_3b"],
    #             config["dataset"]["ratio_4b"],
    #             config["dataset"]["signal_ratio"],
    #             config["dataset"]["signal_filename"],
    #             config["dataset"]["seed"],
    #         ),
    #         "train_seed": ensemble_seed,
    #         "model_seed": ensemble_seed,
    #         "data_seed": ensemble_seed,
    #     }
    # )

    # Step 3 config
    config["CR_fvt"]["train_seed"] = ensemble_seed
    config["CR_fvt"]["model_seed"] = ensemble_seed
    config["CR_fvt"]["data_seed"] = ensemble_seed
    previous_step_experiment_name = "smeared_fvt_training_ensemble"
    config["previous_step_experiment_name"] = previous_step_experiment_name
    config["signal_region"]["SR_stats_hashes"] = find_and_check_hashes_have_same_ms(
        {
            "experiment_name": previous_step_experiment_name,
            "aux_info_step": 2,
            "dataset": check_dataset_conditions(
                config["dataset"]["n_3b"],
                config["dataset"]["ratio_4b"],
                config["dataset"]["signal_ratio"],
                config["dataset"]["signal_filename"],
                config["dataset"]["seed"],
            ),
        }
    )
    config["signal_region"]["ensemble_mode"] = "max"
    # config["signal_region"]["stats_type"] = "smeared"
    config["signal_region"]["stats_type"] = "fvt"
    config["CR_fvt"]["train_seed"] = ensemble_seed
    config["CR_fvt"]["model_seed"] = ensemble_seed
    config["CR_fvt"]["data_seed"] = ensemble_seed

    # Step 3 config
    # config["CR_fvt"]["train_seed"] = seed
    # config["CR_fvt"]["model_seed"] = seed
    # config["CR_fvt"]["data_seed"] = seed

    # Step 3 config
    # config["CR_fvt"]["train_seed"] = seed
    # config["CR_fvt"]["model_seed"] = seed
    # config["CR_fvt"]["data_seed"] = seed
    # additional config
    # config["previous_step_experiment_name"] = "smeared_fvt_training_repr_norm"
    # config["CR_fvt"]["repr_norm"] = True

    try:
        with open(f"../configs/tmp/{NEW_CONFIG_FILENAME}", "w") as f:
            yaml.dump(config, f)
    except OSError as e:
        logging.error(f"Error writing config file: {e}")
        raise e

    CONFIGS_TO_RUN.append(f"-c configs/tmp/{NEW_CONFIG_FILENAME}")

# Oftentimes we do not change below here
# divide CONFIGS_TO_RUN into N_RUNFILES chunks
CONFIG_CHUNKS = {i: [] for i in range(N_RUNFILES)}
for i, config in enumerate(CONFIGS_TO_RUN):
    CONFIG_CHUNKS[i % N_RUNFILES].append(config)

RUN_PARTITIONS = []
for _ in range(N_RUNFILES):
    RUN_PARTITIONS.append("statds" if np.random.rand() < PROB_STATDS else "phil_condo")

logging.info(
    f"n_tasks (runfiles) {len(CONFIG_CHUNKS)}, max n_tasks per runfile {max([len(chunk) for chunk in CONFIG_CHUNKS.values()])}"
)
logging.info(f"Partitions: {RUN_PARTITIONS}")
# Get input
input("Press Enter to continue...")


for k, configs in CONFIG_CHUNKS.items():
    if len(configs) == 0:
        continue

    RUN_PARTITION = RUN_PARTITIONS[k]
    RUN_FILENAME = f"run_{EXPERIMENT_NAME}_{k}.sh"

    with open("run_multiple_configs_parallel.sh", "r") as f:
        run_script = f.read()

    run_script = run_script.replace(
        'CONFIGS_LIST=""', f'CONFIGS_LIST="{" ".join(configs)}"'
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
