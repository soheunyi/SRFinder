from copy import deepcopy
import tqdm
import yaml
from itertools import product
import numpy as np
import sys
import logging

sys.path.append("..")
from training_info import TrainingInfo
from utils import safe_dict


def get_base_config(BASE_CONFIG_FILENAME: str):
    try:
        with open(f"../configs/{BASE_CONFIG_FILENAME}", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        logging.error(f"Base config file not found: {BASE_CONFIG_FILENAME}")
        raise e
    return config


def find_and_check_hash_unique(hparams_filter: dict, use_cached_metadata: bool = True):
    """Finds a unique training info hash based on a filter.

    Args:
        hparams_filter: Dictionary containing filter criteria.

    Returns:
        The unique hash.

    Raises:
        AssertionError: If more or less than one hash is found.
    """
    hashes = TrainingInfo.find(hparams_filter, use_cached_metadata=use_cached_metadata)
    assert (
        len(hashes) == 1
    ), f"Expected 1 training info, found {len(hashes)} for filter: {hparams_filter}"
    return hashes[0]


def find_and_check_hashes_have_same_ms(
    hparams_filter: dict, use_cached_metadata: bool = True
):
    hashes = TrainingInfo.find(hparams_filter, use_cached_metadata=use_cached_metadata)
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


def step_1_get_configs_to_run(EXPERIMENT_NAME: str, base_config: dict):
    # signal_ratios = [0.0, 0.005, 0.0075, 0.01, 0.02]
    signal_ratios = [0.005, 0.0075, 0.01, 0.02]
    dataset_seeds = range(50)  # start with ten seeds, will be increased to fifty later
    ensemble_seeds = range(15)
    signal_filename = "HH4b_800.h5"

    hparams_filter = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset": lambda x: (
            safe_dict(x, "signal_ratio") in signal_ratios
            and safe_dict(x, "seed") in dataset_seeds
            and safe_dict(x, "signal_filename") == signal_filename
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
    targets.sort(key=lambda x: (x[0], x[1], x[2]))
    logging.info(f"Number of targets: {len(targets)}")

    config_filenames = []
    configs_to_run = []
    postfixs = []
    input(
        f"Confirm that the step is correct: {base_config['step']}, press Enter to continue..."
    )

    for signal_ratio, seed, ensemble_seed in tqdm.tqdm(targets):
        config = deepcopy(base_config)
        postfix = f"{seed}_{signal_ratio}_{ensemble_seed}_{signal_filename}"
        if postfix in postfixs:
            raise ValueError(f"Postfix {postfix} already exists, cannot be duplicated")
        postfixs.append(postfix)

        config["experiment_name"] = EXPERIMENT_NAME
        config["dataset"]["signal_ratio"] = signal_ratio
        config["dataset"]["seed"] = seed
        config["dataset"]["signal_filename"] = signal_filename
        config_filename = f"{EXPERIMENT_NAME}_{postfix}.yml"
        config_filenames.append(config_filename)

        config["base_fvt"]["train_seed"] = ensemble_seed
        config["base_fvt"]["model_seed"] = ensemble_seed
        config["base_fvt"]["data_seed"] = ensemble_seed
        configs_to_run.append(config)

    return config_filenames, configs_to_run


def step_2_get_configs_to_run(EXPERIMENT_NAME: str, base_config: dict):
    ensemble_seeds = range(15)
    dataset_seeds = range(50)  # start with ten seeds, will be increased to fifty later

    # signal_ratios = [0.005, 0.0075, 0.01, 0.02]
    # noise_scales = [0.5, 1.0, 2.0, 3.0]
    # signal_filename = "HH4b_400.h5"
    # base_experiment_name = "base_fvt_training_ensemble_HH4b_400"
    # base_experiment_name = "base_fvt_training_ensemble_HH4b_800"

    # noise_scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    # noise_scales = [0.5, 1.0]
    noise_scales = [2.0, 3.0]
    # base_experiment_name = "base_fvt_training_ensemble"
    signal_ratios = [0.0, 0.005, 0.0075, 0.01, 0.02]
    # signal_filename = "HH4b_picoAOD.h5"

    if EXPERIMENT_NAME == "smeared_fvt_training_ensemble":
        base_experiment_name = "base_fvt_training_ensemble"
        signal_filename = "HH4b_picoAOD.h5"
        signal_ratios = [0.0, 0.005, 0.0075, 0.01, 0.02]
    elif EXPERIMENT_NAME == "smeared_fvt_training_ensemble_HH4b_400":
        base_experiment_name = "base_fvt_training_ensemble_HH4b_400"
        signal_filename = "HH4b_400.h5"
        signal_ratios = [0.005, 0.0075, 0.01, 0.02]
    elif EXPERIMENT_NAME == "smeared_fvt_training_ensemble_HH4b_800":
        base_experiment_name = "base_fvt_training_ensemble_HH4b_800"
        signal_filename = "HH4b_800.h5"
        signal_ratios = [0.005, 0.0075, 0.01, 0.02]
    else:
        raise ValueError(f"Unknown experiment name: {EXPERIMENT_NAME}")

    hparams_filter = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset": lambda x: (
            safe_dict(x, "signal_ratio") in signal_ratios
            and safe_dict(x, "seed") in dataset_seeds
            and safe_dict(x, "signal_filename") == signal_filename
        ),
        "train_seed": lambda x: x in ensemble_seeds,
        "model_seed": lambda x: x in ensemble_seeds,
        "data_seed": lambda x: x in ensemble_seeds,
        "smearing": lambda x: x["noise_scale"] in noise_scales,
    }
    _, hparams = TrainingInfo.find(hparams_filter, return_hparams=True)
    existing = {
        (
            hparam["dataset"]["signal_ratio"],
            hparam["dataset"]["seed"],
            hparam["train_seed"],
            hparam["smearing"]["noise_scale"],
        )
        for hparam in hparams
    }
    targets = set(product(signal_ratios, dataset_seeds, ensemble_seeds, noise_scales))
    targets = list(targets - existing)
    targets.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    logging.info(f"Number of targets: {len(targets)}")

    config_filenames = []
    configs_to_run = []
    postfixs = []
    input(
        f"Confirm that the step is correct: {base_config['step']}, press Enter to continue..."
    )

    for signal_ratio, seed, ensemble_seed, noise_scale in tqdm.tqdm(targets):
        config = deepcopy(base_config)
        postfix = f"{seed}_{signal_ratio}_{ensemble_seed}_{noise_scale}"
        if postfix in postfixs:
            raise ValueError(f"Postfix {postfix} already exists, cannot be duplicated")
        postfixs.append(postfix)
        config_filename = f"{EXPERIMENT_NAME}_{postfix}.yml"
        config_filenames.append(config_filename)

        config["dataset"]["seed"] = seed
        config["dataset"]["signal_ratio"] = signal_ratio
        config["dataset"]["signal_filename"] = signal_filename
        config["experiment_name"] = EXPERIMENT_NAME

        config["smeared_fvt"]["train_seed"] = ensemble_seed
        config["smeared_fvt"]["model_seed"] = ensemble_seed
        config["smeared_fvt"]["data_seed"] = ensemble_seed
        config["smearing"]["noise_scale"] = noise_scale
        config["smearing"]["seed"] = seed
        config["base_experiment_name"] = base_experiment_name
        config["base_experiment_hash"] = find_and_check_hash_unique(
            {
                "experiment_name": base_experiment_name,
                "aux_info_step": 1,
                "dataset": check_dataset_conditions(
                    config["dataset"]["n_3b"],
                    config["dataset"]["ratio_4b"],
                    config["dataset"]["signal_ratio"],
                    config["dataset"]["signal_filename"],
                    config["dataset"]["seed"],
                ),
                "train_seed": ensemble_seed,
                "model_seed": ensemble_seed,
                "data_seed": ensemble_seed,
            }
        )
        configs_to_run.append(config)

    return config_filenames, configs_to_run


def step_3_get_configs_to_run(EXPERIMENT_NAME: str, base_config: dict):
    dataset_seeds = range(50)  # start with ten seeds, will be increased to fifty later
    ensemble_seeds = range(1)
    SR_CR_sizes = [(0.05, 0.95), (0.1, 0.9), (0.15, 0.85), (0.2, 0.8)]
    noise_scales = [0.5, 1.0, 2.0, 3.0, np.inf]
    # noise_scales = [0.5, 1.0, np.inf]
    # noise_scales = [2.0, 3.0]
    # previous_step_experiment_name = "smeared_fvt_training_ensemble_HH4b_400"

    if EXPERIMENT_NAME in [
        "CR_fvt_training_ensemble_max",
        "CR_fvt_training_ensemble_mean",
    ]:
        previous_step_experiment_name = "smeared_fvt_training_ensemble"
    elif EXPERIMENT_NAME in [
        "CR_fvt_training_ensemble_max_HH4b_400",
        "CR_fvt_training_ensemble_mean_HH4b_400",
    ]:
        previous_step_experiment_name = "smeared_fvt_training_ensemble_HH4b_400"
    elif EXPERIMENT_NAME in [
        "CR_fvt_training_ensemble_max_HH4b_800",
        "CR_fvt_training_ensemble_mean_HH4b_800",
    ]:
        previous_step_experiment_name = "smeared_fvt_training_ensemble_HH4b_800"
    else:
        raise ValueError(f"Unknown experiment name: {EXPERIMENT_NAME}")

    if previous_step_experiment_name == "smeared_fvt_training_ensemble":
        signal_ratios = [0.0, 0.005, 0.0075, 0.01, 0.02]
        signal_filename = "HH4b_picoAOD.h5"
    elif previous_step_experiment_name == "smeared_fvt_training_ensemble_HH4b_400":
        signal_ratios = [0.005, 0.0075, 0.01, 0.02]
        signal_filename = "HH4b_400.h5"
    elif previous_step_experiment_name == "smeared_fvt_training_ensemble_HH4b_800":
        signal_ratios = [0.005, 0.0075, 0.01, 0.02]
        signal_filename = "HH4b_800.h5"
    else:
        raise ValueError(
            f"Unknown previous step experiment name: {previous_step_experiment_name}"
        )

    print(f"Previous step experiment name: {previous_step_experiment_name}")
    print(f"Signal filename: {signal_filename}")
    print(f"Signal ratios: {signal_ratios}")
    print(f"Noise scales: {noise_scales}")
    print(f"Dataset seeds: {dataset_seeds}")
    print(f"Ensemble seeds: {ensemble_seeds}")
    print(f"SR_CR_sizes: {SR_CR_sizes}")
    input("Press Enter to continue...")

    metadata = TrainingInfo.load_metadata()

    hparams_filter = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset": lambda x: (
            safe_dict(x, "signal_ratio") in signal_ratios
            and safe_dict(x, "seed") in dataset_seeds
            and safe_dict(x, "signal_filename") == signal_filename
        ),
        "train_seed": lambda x: x in ensemble_seeds,
        "model_seed": lambda x: x in ensemble_seeds,
        "data_seed": lambda x: x in ensemble_seeds,
    }
    _, hparams = TrainingInfo.find(hparams_filter, return_hparams=True)
    print("Existing hashes: ", len(hparams))
    print("Hparams after step 1: ", len(hparams))
    for hparam in hparams:
        if "SR_stats_hashes" not in hparam["signal_region"]:
            print(hparam)
            raise ValueError("SR_stats_hashes not found in signal_region")

        first_hash = hparam["signal_region"]["SR_stats_hashes"][0]
        step_2_hparam = metadata[first_hash]
        if hparam["signal_region"]["stats_type"] == "smeared":
            hparam["noise_scale"] = step_2_hparam["smearing"]["noise_scale"]
        elif hparam["signal_region"]["stats_type"] == "fvt":
            hparam["noise_scale"] = np.inf
        else:
            raise ValueError(
                f"Unknown stats_type: {hparam['signal_region']['stats_type']}"
            )

    print("Hparams after step 2: ", len(hparams))
    existing = {
        (
            hparam["dataset"]["signal_ratio"],
            hparam["dataset"]["seed"],
            hparam["train_seed"],
            hparam["noise_scale"],
            (hparam["signal_region"]["4b_in_SR"], hparam["signal_region"]["4b_in_CR"]),
        )
        for hparam in hparams
    }
    print("Existing hashes: ", len(existing))
    targets = set(
        product(signal_ratios, dataset_seeds, ensemble_seeds, noise_scales, SR_CR_sizes)
    )
    print("Targets: ", len(targets))
    targets = list(targets - existing)
    print("Targets after removing existing: ", len(targets))
    targets.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

    logging.info(f"Number of targets: {len(targets)}")

    config_filenames = []
    configs_to_run = []
    postfixs = []
    input(
        f"Confirm that the step is correct: {base_config['step']}, press Enter to continue..."
    )
    for signal_ratio, seed, ensemble_seed, noise_scale, SR_CR_size in tqdm.tqdm(
        targets
    ):
        postfix = f"{seed}_{signal_ratio}_{ensemble_seed}_{noise_scale}_{SR_CR_size[0]}_{SR_CR_size[1]}"
        if postfix in postfixs:
            raise ValueError(f"Postfix {postfix} already exists, cannot be duplicated")
        postfixs.append(postfix)
        config_filename = f"{EXPERIMENT_NAME}_{postfix}.yml"
        config_filenames.append(config_filename)

        config = deepcopy(base_config)
        config["experiment_name"] = EXPERIMENT_NAME
        config["dataset"]["signal_ratio"] = signal_ratio
        config["dataset"]["seed"] = seed
        config["dataset"]["signal_filename"] = signal_filename

        assert "step" in config, "Step must be specified in the config file"

        config["CR_fvt"]["train_seed"] = ensemble_seed
        config["CR_fvt"]["model_seed"] = ensemble_seed
        config["CR_fvt"]["data_seed"] = ensemble_seed
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
                "smearing": lambda x: (
                    x["noise_scale"]
                    == (
                        noise_scale
                        if noise_scale != np.inf
                        # use 1.0 for stats_type=fvt, which does not use smearing.
                        # noise_scale does not matter for this case.
                        else 1.0
                    )
                ),
            },
        )
        assert (
            len(config["signal_region"]["SR_stats_hashes"]) == 15
        ), f"Expected 15 SR stats hashes, got {len(config['signal_region']['SR_stats_hashes'])}"

        if EXPERIMENT_NAME in [
            "CR_fvt_training_ensemble_max",
            "CR_fvt_training_ensemble_max_HH4b_400",
            "CR_fvt_training_ensemble_max_HH4b_800",
        ]:
            config["signal_region"]["ensemble_mode"] = "max"
        elif EXPERIMENT_NAME in [
            "CR_fvt_training_ensemble_mean",
            "CR_fvt_training_ensemble_mean_HH4b_400",
            "CR_fvt_training_ensemble_mean_HH4b_800",
        ]:
            config["signal_region"]["ensemble_mode"] = "mean"
        else:
            raise ValueError(f"Unknown experiment name: {EXPERIMENT_NAME}")

        if noise_scale == np.inf:
            config["signal_region"]["stats_type"] = "fvt"
        else:
            config["signal_region"]["stats_type"] = "smeared"

        config["signal_region"]["4b_in_SR"] = SR_CR_size[0]
        config["signal_region"]["4b_in_CR"] = SR_CR_size[1]

        config["CR_fvt"]["train_seed"] = ensemble_seed
        config["CR_fvt"]["model_seed"] = ensemble_seed
        config["CR_fvt"]["data_seed"] = ensemble_seed

        configs_to_run.append(config)

    return config_filenames, configs_to_run


def step_4_get_configs_to_run(EXPERIMENT_NAME: str, base_config: dict):
    signal_ratios = [0.0, 0.005, 0.0075, 0.01, 0.02]
    dataset_seeds = range(50)
    SR_CR_sizes = [(0.05, 0.95), (0.1, 0.9), (0.15, 0.85), (0.2, 0.8)]
    mi_train_seed = 0
    batch_size = 256
    previous_step_experiment_name = "CR_fvt_training_ensemble_max_smeared"

    config_filenames = []
    configs_to_run = []
    postfixs = []
    sr_seed_srcrsizes = list(product(signal_ratios, dataset_seeds, SR_CR_sizes))
    for signal_ratio, seed, SR_CR_size in sr_seed_srcrsizes:
        config = deepcopy(base_config)
        config["mi_test_fvt"]["dataloader"]["batch_size"] = batch_size
        config["mi_test_fvt"]["fit_batch_size"] = batch_size
        config["mi_test_fvt"]["resample"] = True

        SR_size, CR_size = SR_CR_size
        postfix = f"{seed}_{signal_ratio}_{SR_size}_{CR_size}"
        if postfix in postfixs:
            raise ValueError(f"Postfix {postfix} already exists, cannot be duplicated")
        postfixs.append(postfix)
        config_filename = f"{EXPERIMENT_NAME}_{postfix}.yml"
        config_filenames.append(config_filename)

        config["dataset"]["seed"] = seed
        config["dataset"]["signal_ratio"] = signal_ratio
        config["experiment_name"] = EXPERIMENT_NAME

        config["previous_step_experiment_name"] = previous_step_experiment_name
        config["mi_test_fvt"]["train_seed"] = mi_train_seed
        config["mi_test_fvt"]["model_seed"] = mi_train_seed
        config["mi_test_fvt"]["data_seed"] = mi_train_seed
        config["CR_fvt_hash"] = find_and_check_hash_unique(
            {
                "experiment_name": previous_step_experiment_name,
                "aux_info_step": 3,
                "dataset": check_dataset_conditions(
                    config["dataset"]["n_3b"],
                    config["dataset"]["ratio_4b"],
                    config["dataset"]["signal_ratio"],
                    config["dataset"]["signal_filename"],
                    config["dataset"]["seed"],
                ),
                "signal_region": lambda x: (
                    x["4b_in_SR"] == SR_size and x["4b_in_CR"] == CR_size
                ),
            }
        )

        configs_to_run.append(config)

    return config_filenames, configs_to_run


def write_and_get_configs_to_run(
    STEP: int, EXPERIMENT_NAME: str, BASE_CONFIG_FILENAME: str
):
    base_config = get_base_config(BASE_CONFIG_FILENAME)
    if STEP == 1:
        config_filenames, configs_to_run = step_1_get_configs_to_run(
            EXPERIMENT_NAME, base_config
        )
    elif STEP == 2:
        config_filenames, configs_to_run = step_2_get_configs_to_run(
            EXPERIMENT_NAME, base_config
        )
    elif STEP == 3:
        config_filenames, configs_to_run = step_3_get_configs_to_run(
            EXPERIMENT_NAME, base_config
        )
    elif STEP == 4:
        config_filenames, configs_to_run = step_4_get_configs_to_run(
            EXPERIMENT_NAME, base_config
        )
    else:
        raise ValueError(f"Unsupported step: {STEP}")

    logging.info(f"Number of configs to run: {len(configs_to_run)}")
    logging.info("Starting to write configs to file...")

    for config_filename, config in tqdm.tqdm(zip(config_filenames, configs_to_run)):
        try:
            with open(f"../configs/tmp/{config_filename}", "w") as f:
                yaml.dump(config, f)
        except OSError as e:
            logging.error(f"Error writing config file: {e}")
            raise e

    CONFIG_STRINGS = [
        f"-c configs/tmp/{config_filename}" for config_filename in config_filenames
    ]
    return CONFIG_STRINGS
