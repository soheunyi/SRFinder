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
from step_3_preprocessing import get_step_3_tinfo_events, check_and_get_CR_fvt_hparams
from training_info import TrainingInfo
from utils import require_keys
from signal_region import get_SR_CR_cut, compute_sr_stats
from constants import FEATURES


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


def routine(config: dict, file_handler: logging.FileHandler | None = None):
    print("Experiment Configuration")
    print(config)
    print("Current Time: ", pd.Timestamp.now())

    CR_fvt_hparams = check_and_get_CR_fvt_hparams(config)
    signal_ratio = CR_fvt_hparams["dataset"]["signal_ratio"]
    SR_stats_hashes = CR_fvt_hparams["signal_region"]["SR_stats_hashes"]

    CR_fvt_tinfo, events_train, events_tst, SR_idx_train, SR_idx = (
        get_step_3_tinfo_events(CR_fvt_hparams)
    )

    CR_fvt_train_dset, CR_fvt_val_dset = CR_fvt_tinfo.fetch_train_val_tensor_datasets(
        FEATURES,
        label="fourTag",
        weight="weight",
        label_dtype=torch.long,
    )

    pl.seed_everything(CR_fvt_hparams["model_seed"])

    if CR_fvt_hparams["model"] == "AttentionClassifier":
        if len(SR_stats_hashes) > 1:
            raise ValueError("AttentionClassifier does not support ensemble mode")
        tinfo_0 = TrainingInfo.load(SR_stats_hashes[0])
        base_encoder_hash = tinfo_0.hparams["encoder_hash"]
        base_fvt_model = TrainingInfo.load(base_encoder_hash).load_trained_model("best")
        q_repr_train = base_fvt_model.q_repr(CR_fvt_train_dset.tensors[0])
        q_repr_val = base_fvt_model.q_repr(CR_fvt_val_dset.tensors[0])
        CR_fvt_train = TensorDataset(
            q_repr_train, CR_fvt_train_dset.tensors[1], CR_fvt_train_dset.tensors[2]
        )
        CR_fvt_val = TensorDataset(
            q_repr_val, CR_fvt_val_dset.tensors[1], CR_fvt_val_dset.tensors[2]
        )
        CR_fvt_model = AttentionClassifier(
            dim_q=CR_fvt_hparams["dim_q"],
            num_classes=2,
            run_name=CR_fvt_tinfo.hash,
            depth=CR_fvt_hparams["depth"],
        )
    elif CR_fvt_hparams["model"] == "FvTClassifier":
        CR_fvt_train = CR_fvt_train_dset
        CR_fvt_val = CR_fvt_val_dset
        CR_fvt_model = FvTClassifier(
            num_classes=2,
            dim_input_jet_features=4,
            dim_dijet_features=CR_fvt_hparams["dim_dijet_features"],
            dim_quadjet_features=CR_fvt_hparams["dim_quadjet_features"],
            run_name=CR_fvt_tinfo.hash,
            depth=CR_fvt_hparams["depth"],
        )
    else:
        raise ValueError(f"Unknown model type: {CR_fvt_hparams['model']}")

    CR_fvt_model.fit(
        CR_fvt_train,
        CR_fvt_val,
        max_epochs=CR_fvt_hparams["max_epochs"],
        train_seed=CR_fvt_hparams["train_seed"],
        save_checkpoint=True,
        callbacks=[],
        tb_log_dir="_".join([config["experiment_name"], str(signal_ratio)]),
        optimizer_config=CR_fvt_hparams["optimizer"],
        lr_scheduler_config=CR_fvt_hparams["lr_scheduler"],
        early_stop_patience=CR_fvt_hparams["early_stop_patience"],
        dataloader_config=CR_fvt_hparams["dataloader"],
        file_handler=file_handler,
    )

    CR_fvt_model.eval()
    CR_fvt_model.to(
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    CR_fvt_model: FvTClassifier

    events_train_SR = events_train[SR_idx_train]
    events_tst_SR = events_tst[SR_idx]
    fvt_scores_train_SR = (
        CR_fvt_model.predict(events_train_SR.X_torch)[:, 1].detach().cpu().numpy()
    )
    fvt_scores_tst_SR = (
        CR_fvt_model.predict(events_tst_SR.X_torch)[:, 1].detach().cpu().numpy()
    )

    CR_fvt_tinfo.update_aux_info(
        description=f"FvT on Control Region",
        step=3,
        fvt_scores_train_SR=fvt_scores_train_SR,
        fvt_scores_tst_SR=fvt_scores_tst_SR,
    )
    CR_fvt_tinfo.save()
    # TrainingInfo.update_metadata()


@click.command()
@click.option("--config", type=str)
def main(config):
    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    routine(config)


if __name__ == "__main__":
    main()
