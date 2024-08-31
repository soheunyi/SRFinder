from itertools import product
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import click
import yaml


from events_data import EventsData
from fvt_classifier import FvTClassifier
from dataset import generate_tt_dataset
from training_info import TrainingInfo


def routine_wrapper(config: dict):
    required_keys = [
        "signal_ratio",
        "dim_dijet_features",
        "dim_quadjet_features",
        "n_3b",
        "n_all4b",
        "test_ratio",
        "val_ratio",
        "batch_size",
        "max_epochs",
        "seed",
        "lr",
        "experiment_name",
        "n_sample_ratio",
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Key {key} is missing in the config")

    routine(
        config["signal_ratio"],
        config["dim_dijet_features"],
        config["dim_quadjet_features"],
        config["n_3b"],
        config["n_all4b"],
        config["test_ratio"],
        config["val_ratio"],
        config["batch_size"],
        config["max_epochs"],
        config["seed"],
        config["lr"],
        config["experiment_name"],
        config["n_sample_ratio"],
    )


def routine(
    signal_ratio,
    dim_dijet_features,
    dim_quadjet_features,
    n_3b,
    n_all4b,
    test_ratio,
    val_ratio,
    batch_size,
    max_epochs,
    seed,
    lr,
    experiment_name,
    n_sample_ratio,
):

    dinfo_train_all, _ = generate_tt_dataset(
        seed,
        n_3b,
        n_all4b,
        signal_ratio,
        test_ratio,
    )

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

    hparams = {
        "signal_ratio": signal_ratio,
        "dim_dijet_features": dim_dijet_features,
        "dim_quadjet_features": dim_quadjet_features,
        "n_3b": n_3b,
        "n_all4b": n_all4b,
        "test_ratio": test_ratio,
        "val_ratio": val_ratio,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "seed": seed,
        "lr": lr,
        "experiment_name": experiment_name,
        "n_sample_ratio": n_sample_ratio,
    }

    num_classes = 2
    dim_input_jet_features = 4

    pl.seed_everything(seed)
    np.random.seed(seed)

    n_train_val = int(n_sample_ratio * len(dinfo_train_all))
    train_val_idx = np.random.choice(
        len(dinfo_train_all),
        n_train_val,
        replace=False,
    )
    n_val = int(val_ratio * n_train_val)

    dinfo_train = dinfo_train_all[train_val_idx[n_val:]]
    dinfo_val = dinfo_train_all[train_val_idx[:n_val]]

    events_train = EventsData.from_dataframe(
        dinfo_train.fetch_data(),
        features,
        name="fvt_train",
    )
    events_val = EventsData.from_dataframe(
        dinfo_val.fetch_data(),
        features,
        name="fvt_val",
    )

    # reduce number of 4b samples to 1/8
    print(
        "4b ratio: ",
        events_train.total_weight_4b / events_train.total_weight,
    )
    print(
        "Signal ratio: ",
        events_train.total_weight_signal / events_train.total_weight_4b,
    )

    events_train.fit_batch_size(batch_size)
    events_val.fit_batch_size(batch_size)

    ###########################################################################################
    ###########################################################################################
    tinfo = TrainingInfo(hparams, dinfo_train, dinfo_val)
    print("Training Hash: ", tinfo.hash)
    tinfo.save()

    model = FvTClassifier(
        num_classes,
        dim_input_jet_features,
        dim_dijet_features,
        dim_quadjet_features,
        run_name=tinfo.hash,
        device=torch.device("cuda:0"),
        lr=lr,
    )

    model.fit(
        events_train.to_tensor_dataset(),
        events_val.to_tensor_dataset(),
        batch_size=batch_size,
        max_epochs=max_epochs,
    )


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

    single_value_keys = ["experiment_name"]
    multiple_values_keys = [
        "signal_ratio",
        "dim_dijet_features",
        "dim_quadjet_features",
        "n_3b",
        "n_all4b",
        "test_ratio",
        "val_ratio",
        "batch_size",
        "max_epochs",
        "seed",
        "lr",
        "n_sample_ratio",
    ]
    yml_parsed = {}

    missing_keys = []

    for k in single_value_keys:
        if k in config:
            yml_parsed[k] = config[k]
        else:
            missing_keys.append(k)

    for k in multiple_values_keys:
        if k in config:
            yml_parsed[k] = listify(config[k])
        else:
            missing_keys.append(k)

    if len(missing_keys) > 0:
        raise ValueError(f"Missing keys in the config: {missing_keys}")

    configs = product(*[yml_parsed[k] for k in multiple_values_keys])
    configs = [dict(zip(multiple_values_keys, c)) for c in configs]
    for c in configs:
        for k in single_value_keys:
            c.update({k: yml_parsed[k]})

    for c in configs:
        routine_wrapper(c)


if __name__ == "__main__":

    main()
