from copy import deepcopy
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import click
import yaml


from fvt_classifier import FvTClassifier
from dataset import MotherSamples
from attention_classifier import AttentionClassifier
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
            "smearing",
            "smeared_fvt",
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
            "encoder_mode",
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

    signal_ratio = config["dataset"]["signal_ratio"]
    n_3b = config["dataset"]["n_3b"]
    ratio_4b = config["dataset"]["ratio_4b"]
    signal_filename = config["dataset"]["signal_filename"]
    seed = config["dataset"]["seed"]

    smeared_fvt_hparams = deepcopy(config["smeared_fvt"])
    smeared_fvt_hparams["experiment_name"] = config["experiment_name"]
    smeared_fvt_hparams["dataset"] = config["dataset"]
    smeared_fvt_hparams["smearing"] = config["smearing"]

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

    # 1. Find and load encoder
    hashes = TrainingInfo.find(
        {
            "dataset": lambda x: (
                x["n_3b"] == n_3b
                and x["ratio_4b"] == ratio_4b
                and x["signal_ratio"] == signal_ratio
                and x["signal_filename"] == signal_filename
                and x["seed"] == seed
            ),
        }
    )
    assert len(hashes) == 1, "Number of training info must be one"
    base_fvt_tinfo = TrainingInfo.load(hashes[0])
    smeared_fvt_hparams["encoder_hash"] = base_fvt_tinfo.hash

    # Shares the same training + val samples
    smeared_fvt_tinfo = TrainingInfo(
        smeared_fvt_hparams,
        ms_hash=base_fvt_tinfo.ms_hash,
        ms_idx=base_fvt_tinfo.ms_idx,
    )

    smeared_fvt_train_dset, smeared_fvt_val_dset = (
        smeared_fvt_tinfo.fetch_train_val_smeared_features(
            features,
            label="fourTag",
            weight="weight",
            label_dtype=torch.long,
        )
    )
    dim_q = smeared_fvt_train_dset.tensors[0].shape[1]
    print(f"dim_q: {dim_q}")
    print(f"Shape of smeared_fvt_train_dset: {smeared_fvt_train_dset.tensors[0].shape}")
    pl.seed_everything(smeared_fvt_hparams["model_seed"])
    smeared_fvt_model = AttentionClassifier(
        dim_q=dim_q,
        num_classes=2,
        run_name=smeared_fvt_tinfo.hash,
        depth=smeared_fvt_hparams["depth"],
    )

    smeared_fvt_model.fit(
        smeared_fvt_train_dset,
        smeared_fvt_val_dset,
        max_epochs=smeared_fvt_hparams["max_epochs"],
        train_seed=smeared_fvt_hparams["train_seed"],
        save_checkpoint=True,
        callbacks=[],
        tb_log_dir="_".join([config["experiment_name"], str(signal_ratio)]),
        optimizer_config=smeared_fvt_hparams["optimizer"],
        lr_scheduler_config=smeared_fvt_hparams["lr_scheduler"],
        early_stop_patience=smeared_fvt_hparams["early_stop_patience"],
        dataloader_config=smeared_fvt_hparams["dataloader"],
    )

    smeared_fvt_tinfo.save()
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
