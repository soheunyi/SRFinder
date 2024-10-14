from copy import deepcopy
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import click
import yaml


from fvt_classifier import FvTClassifier
from dataset import MotherSamples
from training_info import TrainingInfo
from utils import require_keys

###########################################################################################
###########################################################################################
# For each instance of experiment, there would be two FvTClassifier models to be trained:
# 1. base_fvt_model: to learn 3b vs 4b on the mother (training) dataset
# 2. CR_fvt_model: to learn 3b vs 4b on the control region for background estimation
# We require YAML file to specify the hyperparameters separately for each model.
# We will save the trained models in the checkpoints.
# For smear-based SR definition, there is a AttentionClassifier to be trained, but we first
# do not save them in the checkpoints.
###########################################################################################
W_4B_CUT_MIN = 0.001
W_4B_CUT_MAX = 0.999


def routine(config: dict):
    print("Experiment Configuration")
    print(config)
    print("Current Time: ", pd.Timestamp.now())

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

    signal_ratio = config["dataset"]["signal_ratio"]
    n_3b = config["dataset"]["n_3b"]
    ratio_4b = config["dataset"]["ratio_4b"]
    signal_filename = config["dataset"]["signal_filename"]
    seed = config["dataset"]["seed"]
    base_fvt_train_ratio = config["dataset"]["base_fvt_train_ratio"]

    base_fvt_hparams = deepcopy(config["base_fvt"])
    base_fvt_hparams["experiment_name"] = config["experiment_name"]
    base_fvt_hparams["dataset"] = config["dataset"]

    # Define features
    features = [
        "sym_Jet0_pt",
        "sym_Jet1_pt",
        "sym_Jet2_pt",
        "sym_Jet3_pt",
        "sym_Jet0_eta",
        "sym_Jet1_eta",
        "sym_Jet2_eta",
        "sym_Jet3_eta",
        "sym_Jet0_phi",
        "sym_Jet1_phi",
        "sym_Jet2_phi",
        "sym_Jet3_phi",
        "sym_Jet0_m",
        "sym_Jet1_m",
        "sym_Jet2_m",
        "sym_Jet3_m",
    ]

    # Unchangeable configurations
    dim_input_jet_features = 4
    num_classes = 2

    # 1. Find and load the mother dataset
    ms_hparams = {
        "n_3b": n_3b,
        "ratio_4b": ratio_4b,
        "signal_ratio": signal_ratio,
        "signal_filename": signal_filename,
        "seed": seed,
    }
    hashes = MotherSamples.find(ms_hparams)
    assert len(hashes) == 1, "Number of mother samples must be one"
    ms_hash = hashes[0]
    mother_samples = MotherSamples.load(ms_hash)

    # 2. Split the mother dataset into train and test
    # Train -- validation split will be done by TrainingInfoV2

    ms_len = len(mother_samples.scdinfo)
    ms_idx = np.zeros(ms_len, dtype=bool)
    ms_idx[: int(ms_len * base_fvt_train_ratio)] = True
    np.random.seed(seed)
    np.random.shuffle(ms_idx)

    base_fvt_tinfo = TrainingInfo(base_fvt_hparams, ms_hash=ms_hash, ms_idx=ms_idx)
    print("Base FvT Training Hash: ", base_fvt_tinfo.hash)

    base_fvt_train_dset, base_fvt_val_dset = (
        base_fvt_tinfo.fetch_train_val_tensor_datasets(features, "fourTag", "weight")
    )

    model_seed = base_fvt_hparams["model_seed"]
    pl.seed_everything(model_seed)
    base_fvt_model = FvTClassifier(
        num_classes,
        dim_input_jet_features,
        base_fvt_hparams["dim_dijet_features"],
        base_fvt_hparams["dim_quadjet_features"],
        run_name=base_fvt_tinfo.hash,
        device=torch.device("cuda:0"),
        depth=base_fvt_hparams["depth"],
    )

    base_fvt_model.fit(
        base_fvt_train_dset,
        base_fvt_val_dset,
        max_epochs=base_fvt_hparams["max_epochs"],
        train_seed=base_fvt_hparams["train_seed"],
        save_checkpoint=True,
        callbacks=[],
        tb_log_dir="_".join([config["experiment_name"], str(signal_ratio)]),
        optimizer_config=base_fvt_hparams["optimizer"],
        lr_scheduler_config=base_fvt_hparams["lr_scheduler"],
        early_stop_patience=base_fvt_hparams["early_stop_patience"],
        dataloader_config=base_fvt_hparams["dataloader"],
    )

    base_fvt_tinfo.save()
    TrainingInfo.update_metadata()


@click.command()
@click.option("--config", type=str)
def main(config):
    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    routine(config)


if __name__ == "__main__":
    # load base yml file

    main()
