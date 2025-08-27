from copy import deepcopy

import numpy as np
from dataset import MotherSamples
from training_info import TrainingInfo
from utils import require_keys


def get_step_1_tinfo(base_fvt_hparams: dict):
    # 1. Find and load the mother dataset
    signal_ratio = base_fvt_hparams["dataset"]["signal_ratio"]
    n_3b = base_fvt_hparams["dataset"]["n_3b"]
    ratio_4b = base_fvt_hparams["dataset"]["ratio_4b"]
    signal_filename = base_fvt_hparams["dataset"]["signal_filename"]
    seed = base_fvt_hparams["dataset"]["seed"]
    base_fvt_train_ratio = base_fvt_hparams["dataset"]["base_fvt_train_ratio"]

    ms_hparams = {
        "n_3b": n_3b,
        "ratio_4b": ratio_4b,
        "signal_ratio": signal_ratio,
        "signal_filename": signal_filename,
        "seed": seed,
    }
    hashes = MotherSamples.find(ms_hparams, from_metadata=False)
    if len(hashes) == 0:
        raise ValueError(
            "No mother samples found for the given parameters, first save the mother samples with hparams {}".format(
                ms_hparams
            )
        )
    elif len(hashes) > 1:
        raise ValueError(
            "Number of mother samples must be one, instead of {}".format(len(hashes))
        )
    ms_hash = hashes[0]
    mother_samples = MotherSamples.load(ms_hash)

    ms_len = len(mother_samples.scdinfo)
    ms_idx = np.zeros(ms_len, dtype=bool)
    ms_idx[: int(ms_len * base_fvt_train_ratio)] = True
    np.random.seed(seed)
    np.random.shuffle(ms_idx)

    base_fvt_tinfo = TrainingInfo(base_fvt_hparams, ms_hash=ms_hash, ms_idx=ms_idx)

    print("Base FvT Training Hash: ", base_fvt_tinfo.hash)

    return base_fvt_tinfo


def check_and_get_base_fvt_hparams(config: dict):
    require_keys(
        config,
        [
            "experiment_name",
            "dataset",
            "base_fvt",
        ],
    )
    require_keys(
        config["dataset"],
        [
            "signal_filename",
            "signal_ratio",
            "n_3b",
            "ratio_4b",
            "seed",
            "base_fvt_train_ratio",
        ],
    )
    require_keys(
        config["base_fvt"],
        [
            "model",
            "dim_dijet_features",
            "dim_quadjet_features",
            "depth",
            "fit_batch_size",
            "model_seed",
            "train_seed",
            "data_seed",
            "max_epochs",
            "val_ratio",
            "early_stop_patience",
            "optimizer",
            "lr_scheduler",
            "dataloader",
            "repr_norm",
        ],
    )
    require_keys(config["base_fvt"]["optimizer"], ["type", "lr"])
    require_keys(
        config["base_fvt"]["lr_scheduler"],
        ["type", "factor", "threshold", "patience", "cooldown", "min_lr"],
    )
    require_keys(
        config["base_fvt"]["dataloader"],
        ["batch_size", "batch_size_multiplier", "batch_size_milestones"],
    )

    base_fvt_hparams = deepcopy(config["base_fvt"])
    base_fvt_hparams["experiment_name"] = config["experiment_name"]
    base_fvt_hparams["dataset"] = config["dataset"]
    base_fvt_hparams["step"] = 1

    return base_fvt_hparams
