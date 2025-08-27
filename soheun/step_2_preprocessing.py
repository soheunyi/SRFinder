from copy import deepcopy

import numpy as np
from dataset import MotherSamples
from training_info import TrainingInfo
from utils import require_keys


def get_step_2_tinfo(smeared_fvt_hparams: dict):
    base_fvt_tinfo = TrainingInfo.load(smeared_fvt_hparams["encoder_hash"])
    return TrainingInfo(
        smeared_fvt_hparams,
        ms_hash=base_fvt_tinfo.ms_hash,
        ms_idx=base_fvt_tinfo.ms_idx,
    )


def check_and_get_smeared_fvt_hparams(config: dict):
    require_keys(
        config,
        [
            "experiment_name",
            "dataset",
            "smearing",
            "smeared_fvt",
            "base_experiment_name",
            "base_experiment_hash",
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
        ],
    )
    require_keys(
        config["smearing"],
        [
            "noise_scale",
            "seed",
            "hard_cutoff",
            "scale_mode",
        ],
    )
    require_keys(
        config["smeared_fvt"],
        [
            "model",
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
        ],
    )
    require_keys(config["smeared_fvt"]["optimizer"], ["type", "lr"])
    require_keys(
        config["smeared_fvt"]["lr_scheduler"],
        ["type", "factor", "threshold", "patience", "cooldown", "min_lr"],
    )
    require_keys(
        config["smeared_fvt"]["dataloader"],
        ["batch_size", "batch_size_multiplier", "batch_size_milestones"],
    )

    smeared_fvt_hparams = deepcopy(config["smeared_fvt"])
    smeared_fvt_hparams["experiment_name"] = config["experiment_name"]
    smeared_fvt_hparams["dataset"] = config["dataset"]
    smeared_fvt_hparams["smearing"] = config["smearing"]
    smeared_fvt_hparams["step"] = 2

    signal_ratio = config["dataset"]["signal_ratio"]
    n_3b = config["dataset"]["n_3b"]
    ratio_4b = config["dataset"]["ratio_4b"]
    signal_filename = config["dataset"]["signal_filename"]
    seed = config["dataset"]["seed"]

    if config["base_experiment_hash"] is None:
        raise ValueError("base_experiment_hash is not set")
    base_fvt_hash = config["base_experiment_hash"]
    base_fvt_tinfo = TrainingInfo.load(base_fvt_hash)
    assert base_fvt_tinfo.hparams["experiment_name"] == config["base_experiment_name"]
    assert base_fvt_tinfo.hparams["model"] == "FvTClassifier"
    assert base_fvt_tinfo.aux_info["step"] == 1
    assert base_fvt_tinfo.hparams["dataset"]["n_3b"] == n_3b
    assert base_fvt_tinfo.hparams["dataset"]["ratio_4b"] == ratio_4b
    assert base_fvt_tinfo.hparams["dataset"]["signal_ratio"] == signal_ratio
    assert base_fvt_tinfo.hparams["dataset"]["signal_filename"] == signal_filename
    assert base_fvt_tinfo.hparams["dataset"]["seed"] == seed

    smeared_fvt_hparams["encoder_hash"] = base_fvt_tinfo.hash

    return smeared_fvt_hparams
