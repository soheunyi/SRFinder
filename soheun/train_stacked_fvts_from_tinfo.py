from copy import deepcopy
import logging
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import click
import yaml
from torch.utils.data import TensorDataset


from attention_classifier import AttentionClassifier
from dataset import MotherSamples
from events_data import EventsData, events_from_scdinfo
from fvt_classifier import FvTClassifier
from stacked_fvt import StackedFvTClassifier
from training_info import TrainingInfo
from utils import require_keys, validate_consistent_hparams
from signal_region import get_SR_CR_cut, compute_sr_stats
from constants import FEATURES


###########################################################################################
###########################################################################################
# Get multiple TrainingInfo objects and train FvTClassifier models based on them
###########################################################################################


def train_stacked_fvt(
    tinfos: list[TrainingInfo],
    file_handler: logging.FileHandler | None = None,
):

    critical_hparams = [
        "experiment_name",
        "step",
        "model",
        "dim_dijet_features",
        "dim_quadjet_features",
        "depth.encoder",
        "depth.decoder",
        "fit_batch_size",
        "model_seed",
        "train_seed",
        "data_seed",
        "max_epochs",
        "val_ratio",
        "early_stop_patience",
        "optimizer.type",
        "optimizer.lr",
        "lr_scheduler.type",
        "lr_scheduler.factor",
        "lr_scheduler.threshold",
        "lr_scheduler.patience",
        "lr_scheduler.cooldown",
        "lr_scheduler.min_lr",
        "dataloader.batch_size",
        "dataloader.batch_size_multiplier",
        "dataloader.batch_size_milestones",
        "encoder_mode",
        "repr_norm",
    ]

    num_stacks = len(tinfos)

    # model_seed, train_seed, data_seed should be the same for all tinfos
    consistent, mismatches = validate_consistent_hparams(
        [tinfo.hparams for tinfo in tinfos], critical_hparams
    )
    if not consistent:
        raise ValueError(f"Critical hyperparameters are not consistent: {mismatches}")

    model_seed = tinfos[0].hparams["model_seed"]
    dim_dijet_features = tinfos[0].hparams["dim_dijet_features"]
    dim_quadjet_features = tinfos[0].hparams["dim_quadjet_features"]
    depth = tinfos[0].hparams["depth"]
    repr_norm = tinfos[0].hparams["repr_norm"]
    run_names = [tinfo.hash for tinfo in tinfos]

    stacked_hparams = {
        "num_stacks": num_stacks,  # Use actual number of seeds
        "num_classes": 2,
        "dim_input_jet_features": 4,
        "dim_dijet_features": dim_dijet_features,
        "dim_quadjet_features": dim_quadjet_features,
        "run_names": run_names,
        "stacked_run_name": "_".join([run_names[0], str(num_stacks)]),
        "depth": depth,
        "repr_norm": repr_norm,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    train_datasets = []
    val_datasets = []
    for tinfo in tinfos:
        train_dset, val_dset = tinfo.fetch_train_val_tensor_datasets(
            FEATURES, "fourTag", "weight"
        )
        train_datasets.append(train_dset)
        val_datasets.append(val_dset)
    max_epochs = tinfos[0].hparams["max_epochs"]
    train_seed = tinfos[0].hparams["train_seed"]
    experiment_name = tinfos[0].hparams["experiment_name"]
    step = tinfos[0].hparams["step"]
    optimizer_config = tinfos[0].hparams["optimizer"]
    lr_scheduler_config = tinfos[0].hparams["lr_scheduler"]
    early_stop_patience = tinfos[0].hparams["early_stop_patience"]
    dataloader_config = tinfos[0].hparams["dataloader"]
    dataloader_config["num_workers"] = 8

    fit_args = {
        "train_datasets": train_datasets,
        "val_datasets": val_datasets,
        "max_epochs": max_epochs,
        "train_seed": train_seed,
        "save_checkpoint": True,
        "callbacks": [],
        "tb_log_dir": "_".join([experiment_name, stacked_hparams["stacked_run_name"]]),
        "optimizer_config": optimizer_config,
        "lr_scheduler_config": lr_scheduler_config,
        "early_stop_patience": early_stop_patience,
        "dataloader_config": dataloader_config,
        "file_handler": file_handler,
    }

    pl.seed_everything(model_seed)
    stacked_model = StackedFvTClassifier(**stacked_hparams)
    stacked_model.fit(**fit_args)

    stacked_model.eval()
    stacked_model.to(
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    stacked_model: StackedFvTClassifier

    return stacked_model
