from copy import deepcopy
from itertools import product
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import click
import yaml


from events_data import EventsData
from fvt_classifier import FvTClassifier
from dataset import SCDatasetInfo, generate_mother_dataset, split_scdinfo
from signal_region import get_SR_stats
from training_info import TrainingInfoV2
from tst_info import TSTInfo


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


def require_keys(config: dict, keys: list):
    for key in keys:
        if key not in config:
            raise ValueError(f"Key {key} is missing in the config")


def routine(config: dict):
    require_keys(
        config,
        [
            "signal_ratio",
            "n_3b",
            "ratio_4b",
            "experiment_name",
            "signal_filename",
            "seed",
            "base_fvt_train_ratio",
            "SR_train_ratio",
            "SRCR",
            "base_fvt",
            "CR_fvt",
        ],
    )
    for config_inner in [config["base_fvt"], config["CR_fvt"]]:
        require_keys(
            config_inner,
            [
                "dim_dijet_features",
                "dim_quadjet_features",
                "batch_size",
                "max_epochs",
                "data_seed",
                "train_seed",
                "val_ratio",
                "lr",
            ],
        )

    require_keys(config["SRCR"], ["method", "4b_in_CR", "4b_in_SR"])

    signal_ratio = config["signal_ratio"]
    n_3b = config["n_3b"]
    ratio_4b = config["ratio_4b"]
    experiment_name = config["experiment_name"]
    signal_filename = config["signal_filename"]
    seed = config["seed"]

    base_fvt_hparams = deepcopy(config["base_fvt"])
    base_fvt_hparams["experiment_hparams"] = deepcopy(config)

    CR_fvt_hparams = deepcopy(config["CR_fvt"])
    CR_fvt_hparams["experiment_hparams"] = deepcopy(config)

    SRCR_hparams = config["SRCR"]

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

    # 1. Generate the mother dataset
    scdinfo_all, df_all = generate_mother_dataset(
        n_3b=n_3b,
        ratio_4b=ratio_4b,
        signal_ratio=signal_ratio,
        signal_filename=signal_filename,
        seed=seed,
    )

    # 2. Split the mother dataset into train and test
    # Train -- validation split will be done by TrainingInfoV2

    scdinfo_base_fvt, scdinfo_not_base_fvt = split_scdinfo(
        scdinfo_all, config["base_fvt_train_ratio"], seed
    )

    pl.seed_everything(seed)
    np.random.seed(seed)

    base_fvt_tinfo = TrainingInfoV2(base_fvt_hparams, scdinfo_base_fvt)
    print("Base FvT Training Hash: ", base_fvt_tinfo.hash)
    base_fvt_tinfo.save()

    base_fvt_train_dset, base_fvt_val_dset = (
        base_fvt_tinfo.fetch_train_val_tensor_datasets(features, "fourTag", "weight")
    )

    base_fvt_model = FvTClassifier(
        num_classes,
        dim_input_jet_features,
        base_fvt_hparams["dim_dijet_features"],
        base_fvt_hparams["dim_quadjet_features"],
        run_name=base_fvt_tinfo.hash,
        device=torch.device("cuda:0"),
        lr=base_fvt_hparams["lr"],
    )

    base_fvt_model.fit(
        base_fvt_train_dset,
        base_fvt_val_dset,
        batch_size=base_fvt_hparams["batch_size"],
        max_epochs=base_fvt_hparams["max_epochs"],
        train_seed=base_fvt_hparams["train_seed"],
    )
    base_fvt_model.eval()

    # Get control region

    scdinfo_SR_train, scdinfo_tst = split_scdinfo(
        scdinfo_not_base_fvt, config["SR_train_ratio"], seed
    )

    # Should be modified: get_SR_stats should not have an access to "signal"
    # But initiation of EventsData requires "signal" in the dataframe...
    events_SR_train = events_from_scdinfo(scdinfo_SR_train, features, signal_filename)
    events_tst = events_from_scdinfo(scdinfo_tst, features, signal_filename)

    if SRCR_hparams["method"] == "smearing":
        require_keys(SRCR_hparams, ["noise_scale"])
        SR_stats = get_SR_stats(
            events_tst,
            fvt_hash=base_fvt_tinfo.hash,
            method=SRCR_hparams["method"],
            events_SR_train=events_SR_train,
            noise_scale=SRCR_hparams["noise_scale"],
        )

    elif SRCR_hparams["method"] == "density_peak":
        require_keys(SRCR_hparams, ["peak_pct"])
        SR_stats = get_SR_stats(
            events_tst,
            fvt_hash=base_fvt_tinfo.hash,
            method=SRCR_hparams["method"],
            events_SR_train=events_SR_train,
            peak_pct=SRCR_hparams["peak_pct"],
        )

    elif SRCR_hparams["method"] == "fvt":
        SR_stats = get_SR_stats(
            events_tst,
            fvt_hash=base_fvt_tinfo.hash,
            method=SRCR_hparams["method"],
            events_SR_train=events_SR_train,
        )
    else:
        raise ValueError("Method {} not recognized".format(SRCR_hparams["method"]))

    SR_stats_argsort = np.argsort(SR_stats)[::-1]
    SR_stats_sorted = SR_stats[SR_stats_argsort]

    weights = events_tst.weights[SR_stats_argsort]
    is_4b = events_tst.is_4b[SR_stats_argsort]
    cumul_4b_ratio = np.cumsum(weights * is_4b) / np.sum(weights * is_4b)

    SR_cut = SR_stats_sorted[np.argmin(cumul_4b_ratio < SRCR_hparams["4b_in_SR"])]
    CR_cut = SR_stats_sorted[
        np.argmin(cumul_4b_ratio < SRCR_hparams["4b_in_CR"] + SRCR_hparams["4b_in_SR"])
    ]

    df_tst = scdinfo_tst.fetch_data()
    assert np.all(events_tst.X == df_tst.loc[:, features].values)

    SR_idx = SR_stats >= SR_cut
    events_tst_SR = events_tst[SR_idx]
    scdinfo_tst_SR = scdinfo_tst[SR_idx]

    CR_idx = (SR_stats >= CR_cut) & (SR_stats < SR_cut)
    events_tst_CR = events_tst[CR_idx]
    scdinfo_tst_CR = scdinfo_tst[CR_idx]

    CR_fvt_tinfo = TrainingInfoV2(CR_fvt_hparams, scdinfo_tst_CR)
    print("CR FvT Training Hash: ", CR_fvt_tinfo.hash)
    CR_fvt_tinfo.save()

    CR_fvt_train_dset, CR_fvt_val_dset = CR_fvt_tinfo.fetch_train_val_tensor_datasets(
        features, "fourTag", "weight"
    )

    CR_fvt_model = FvTClassifier(
        num_classes,
        dim_input_jet_features,
        CR_fvt_hparams["dim_dijet_features"],
        CR_fvt_hparams["dim_quadjet_features"],
        run_name=CR_fvt_tinfo.hash,
        device=torch.device("cuda:0"),
        lr=CR_fvt_hparams["lr"],
    )

    CR_fvt_model.fit(
        CR_fvt_train_dset,
        CR_fvt_val_dset,
        batch_size=CR_fvt_hparams["batch_size"],
        max_epochs=CR_fvt_hparams["max_epochs"],
        train_seed=CR_fvt_hparams["train_seed"],
    )
    CR_fvt_model.eval()

    cond_prob_4b_est = (
        CR_fvt_model.predict(events_tst_SR.X_torch)[:, 1].detach().cpu().numpy()
    )
    reweights = ratio_4b * cond_prob_4b_est / ((1 - ratio_4b) * (1 - cond_prob_4b_est))
    events_tst_SR.reweight(
        np.where(
            events_tst_SR.is_4b,
            events_tst_SR.weights,
            events_tst_SR.weights * reweights,
        )
    )

    tst_info = TSTInfo(
        hparams=config,
        scdinfo_tst=scdinfo_tst,
        SR_stats=SR_stats,
        SR_cut=SR_cut,
        CR_cut=CR_cut,
        base_fvt_tinfo_hash=base_fvt_tinfo.hash,
        CR_fvt_tinfo_hash=CR_fvt_tinfo.hash,
    )
    print("Two Sample Test Hash: ", tst_info.hash)
    tst_info.save()


def get_is_signal(scdinfo: SCDatasetInfo, signal_filename: str):
    # Now show the answer
    is_signals = []
    for file, file_len in zip(scdinfo.files, scdinfo.get_file_lengths()):
        is_signals.append(
            np.full(file_len, True)
            if file.name == signal_filename
            else np.full(file_len, False)
        )
    is_signal = np.concatenate(is_signals)
    return is_signal


def events_from_scdinfo(scdinfo: SCDatasetInfo, features: list, signal_filename: str):
    df = scdinfo.fetch_data()
    df["signal"] = get_is_signal(scdinfo, signal_filename)
    events = EventsData.from_dataframe(df, features)

    return events


def listify(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


@click.command()
@click.option("--config", type=str)
def main(config):
    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    routine(config)


if __name__ == "__main__":
    # load base yml file

    main()
