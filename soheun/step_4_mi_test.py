from copy import deepcopy
import logging
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import click
import yaml

from dataset import MotherSamples
from events_data import EventsData, get_is_signal
from fvt_classifier import FvTClassifier
from training_info import TrainingInfo
from utils import require_keys, select_random_true_elements
from signal_region import get_SR_CR_cut, compute_sr_stats


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

    require_keys(
        config,
        [
            "experiment_name",
            "step",
            "CR_fvt_hash",
            "previous_step_experiment_name",
            "dataset",
            "mi_test_dataset",
            "mi_test_fvt",
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
        config["mi_test_dataset"],
        [
            "test_ratio",
            "data_seed",
        ],
    )
    require_keys(
        config["mi_test_fvt"],
        [
            "model",
            "depth",
            "fit_batch_size",
            "model_seed",
            "train_seed",
            "resample",
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
    require_keys(config["mi_test_fvt"]["optimizer"], ["type", "lr"])
    require_keys(
        config["mi_test_fvt"]["lr_scheduler"],
        ["type"],
    )
    if config["mi_test_fvt"]["lr_scheduler"]["type"] == "ReduceLROnPlateau":
        require_keys(
            config["mi_test_fvt"]["lr_scheduler"],
            ["factor", "threshold", "patience", "cooldown", "min_lr"],
        )
    if config["mi_test_fvt"]["optimizer"]["type"] == "AdamWScheduleFree":
        require_keys(
            config["mi_test_fvt"]["optimizer"],
            ["warmup_steps"],
        )
    require_keys(
        config["mi_test_fvt"]["dataloader"],
        ["batch_size", "batch_size_multiplier", "batch_size_milestones"],
    )

    if config["mi_test_fvt"]["model"] == "FvTClassifier":
        require_keys(
            config["mi_test_fvt"],
            ["dim_dijet_features", "dim_quadjet_features"],
        )
        assert isinstance(
            config["mi_test_fvt"]["depth"], dict
        ), "depth must be a dictionary"
        require_keys(
            config["mi_test_fvt"]["depth"],
            ["encoder", "decoder"],
        )
    elif config["mi_test_fvt"]["model"] == "AttentionClassifier":
        require_keys(
            config["mi_test_fvt"],
            ["dim_q"],
        )
    else:
        raise ValueError(f"Unknown model type: {config['mi_test_fvt']['model']}")

    if config["mi_test_fvt"]["model"] != "FvTClassifier":
        assert False, "Only FvTClassifier is supported for model-independent test"

    mi_test_fvt_hparams: dict = deepcopy(config["mi_test_fvt"])
    for key in config.keys():
        if key == "mi_test_fvt":
            continue
        elif key in mi_test_fvt_hparams.keys():
            raise ValueError(f"Key {key} is already in mi_test_fvt_hparams")
        else:
            mi_test_fvt_hparams[key] = config[key]

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

    # Load CR FvT
    CR_fvt_tinfo = TrainingInfo.load(config["CR_fvt_hash"])
    SR_stats_hashes = CR_fvt_tinfo.hparams["signal_region"]["SR_stats_hashes"]
    ensemble_mode = CR_fvt_tinfo.hparams["signal_region"]["ensemble_mode"]
    stats_type = CR_fvt_tinfo.hparams["signal_region"]["stats_type"]
    signal_filename = CR_fvt_tinfo.hparams["dataset"]["signal_filename"]

    ms_idx = TrainingInfo.load(SR_stats_hashes[0]).ms_idx
    msamples = MotherSamples.load(CR_fvt_tinfo.ms_hash)
    train_scdinfo = msamples.scdinfo[ms_idx]
    tst_scdinfo = msamples.scdinfo[~ms_idx]

    df_train = train_scdinfo.fetch_data()
    df_train["signal"] = get_is_signal(train_scdinfo, signal_filename)
    events_train = EventsData.from_dataframe(df_train, features)

    df_tst = tst_scdinfo.fetch_data()
    df_tst["signal"] = get_is_signal(tst_scdinfo, signal_filename)
    events_tst = EventsData.from_dataframe(df_tst, features)

    SR_stats_train, SR_stats_tst = compute_sr_stats(
        SR_stats_hashes,
        signal_filename,
        ensemble_mode,
        stats_type,
    )

    SR_cut, _ = get_SR_CR_cut(
        SR_stats_train, events_train, CR_fvt_tinfo.hparams["signal_region"]
    )
    SR_idx = SR_stats_tst >= SR_cut

    CR_fvt_model = CR_fvt_tinfo.load_trained_model("best")
    CR_fvt_model: FvTClassifier
    CR_fvt_model.eval()

    def reweighting_fn(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        fvt_scores = CR_fvt_model.predict(X).detach().cpu()[:, 1]
        return torch.where(y == 0, fvt_scores / (1 - fvt_scores), 1.0)

    resample_SR = config["mi_test_fvt"]["resample"]

    if resample_SR:
        events_tst_SR = events_tst[SR_idx].clone()
        reweights_tst_SR = (
            reweighting_fn(events_tst_SR.X_torch, events_tst_SR.is_4b_torch)
            .detach()
            .cpu()
            .numpy()
        )
        events_tst_SR.reweight(reweights_tst_SR * events_tst_SR.weights)

        tst_SR_idx_bool = np.zeros_like(ms_idx, dtype=bool)
        tst_idx = np.where(~ms_idx)[0]
        tst_SR_idx_bool[tst_idx[SR_idx]] = True

        mi_test_fvt_tinfo = TrainingInfo(
            mi_test_fvt_hparams,
            ms_hash=msamples.hash,
            ms_idx=tst_SR_idx_bool,
        )

        train_ratio = 1 - config["mi_test_dataset"]["test_ratio"]
        mi_test_fvt_train_dset, mi_test_fvt_val_dset = (
            mi_test_fvt_tinfo.fetch_train_val_tensor_datasets_with_resampling(
                n_samples=int(len(events_tst_SR) * train_ratio),
                features=features,
                label="fourTag",
                weight="weight",
                label_dtype=torch.long,
                reweighting_fn=reweighting_fn,
            )
        )

    else:
        SR_3b_idx = events_tst.is_3b & SR_idx
        SR_4b_idx = events_tst.is_4b & SR_idx
        # select test_ratio of the events in the SR for 3b and 4b
        SR_3b_test_idx = select_random_true_elements(
            SR_3b_idx,
            config["mi_test_dataset"]["test_ratio"],
            config["mi_test_dataset"]["data_seed"],
        )
        SR_4b_test_idx = select_random_true_elements(
            SR_4b_idx,
            config["mi_test_dataset"]["test_ratio"],
            config["mi_test_dataset"]["data_seed"],
        )
        SR_test_idx = SR_3b_test_idx | SR_4b_test_idx

        tst_idx_int = np.where(~ms_idx)[0]
        tst_SR_idx_bool = np.zeros_like(ms_idx, dtype=bool)
        tst_SR_idx_bool[tst_idx_int[SR_idx]] = True

        tst_SR_test_idx_bool = np.zeros_like(ms_idx, dtype=bool)
        tst_SR_test_idx_bool[tst_idx_int[SR_test_idx]] = True

        tst_SR_train_idx_bool = tst_SR_idx_bool & ~tst_SR_test_idx_bool

        assert np.all(tst_SR_idx_bool == tst_SR_test_idx_bool | tst_SR_train_idx_bool)
        assert np.all(tst_SR_test_idx_bool & tst_SR_train_idx_bool == False)
        assert (
            len(ms_idx)
            == len(tst_SR_idx_bool)
            == len(tst_SR_test_idx_bool)
            == len(tst_SR_train_idx_bool)
        )

        mi_test_fvt_tinfo = TrainingInfo(
            mi_test_fvt_hparams,
            ms_hash=msamples.hash,
            ms_idx=tst_SR_train_idx_bool,
        )

        mi_test_fvt_train_dset, mi_test_fvt_val_dset = (
            mi_test_fvt_tinfo.fetch_train_val_tensor_datasets(
                features,
                label="fourTag",
                weight="weight",
                label_dtype=torch.long,
                reweighting_fn=reweighting_fn,
            )
        )

    pl.seed_everything(mi_test_fvt_hparams["model_seed"])

    mi_test_fvt_model = FvTClassifier(
        num_classes=2,
        dim_input_jet_features=4,
        dim_dijet_features=mi_test_fvt_hparams["dim_dijet_features"],
        dim_quadjet_features=mi_test_fvt_hparams["dim_quadjet_features"],
        run_name=mi_test_fvt_tinfo.hash,
        depth=mi_test_fvt_hparams["depth"],
    )

    mi_test_fvt_model.fit(
        mi_test_fvt_train_dset,
        mi_test_fvt_val_dset,
        max_epochs=mi_test_fvt_hparams["max_epochs"],
        train_seed=mi_test_fvt_hparams["train_seed"],
        save_checkpoint=True,
        callbacks=[],
        tb_log_dir=config["experiment_name"],
        optimizer_config=mi_test_fvt_hparams["optimizer"],
        lr_scheduler_config=mi_test_fvt_hparams["lr_scheduler"],
        early_stop_patience=mi_test_fvt_hparams["early_stop_patience"],
        dataloader_config=mi_test_fvt_hparams["dataloader"],
        file_handler=file_handler,
    )

    mi_test_fvt_model.eval()
    mi_test_fvt_model.to(torch.device("cuda"))

    file_handler.stream.write(f"Finished training model {mi_test_fvt_tinfo.hash}\n")
    file_handler.stream.write(f"Current Time: {pd.Timestamp.now()}\n")
    file_handler.stream.flush()

    if resample_SR:
        # compute fvt scores for all SR events
        events_tst_SR = events_tst[SR_idx]
        fvt_scores_tst_SR = (
            mi_test_fvt_model.predict(events_tst_SR.X_torch)
            .detach()
            .cpu()
            .numpy()[:, 1]
        )
        reweights_tst_SR = (
            reweighting_fn(events_tst_SR.X_torch, events_tst_SR.is_4b_torch)
            .detach()
            .cpu()
            .numpy()
        )
        y_train = [v[1].item() for v in mi_test_fvt_train_dset]
        y_val = [v[1].item() for v in mi_test_fvt_val_dset]
        pi = np.mean(y_train + y_val)
        mi_test_fvt_tinfo.update_aux_info(
            description=f"Model independent test for {config['CR_fvt_hash']}",
            step=config["step"],
            fvt_scores_tst_SR=fvt_scores_tst_SR,
            reweights_tst_SR=reweights_tst_SR,
            pi=pi,
        )

    else:
        # compute fvt scores for the test dataset
        events_tst_SR_test = events_tst[SR_test_idx]
        events_tst_SR_train = events_tst[SR_idx & ~SR_test_idx]
        pi = events_tst_SR_train.total_weight_4b / events_tst_SR_train.total_weight
        fvt_scores_tst_SR_test = (
            mi_test_fvt_model.predict(events_tst_SR_test.X_torch)
            .detach()
            .cpu()
            .numpy()[:, 1]
        )
        reweights_tst_SR_test = (
            reweighting_fn(events_tst_SR_test.X_torch, events_tst_SR_test.is_4b_torch)
            .detach()
            .cpu()
            .numpy()
        )
        reweights_tst_SR_train = (
            reweighting_fn(events_tst_SR_train.X_torch, events_tst_SR_train.is_4b_torch)
            .detach()
            .cpu()
            .numpy()
        )
        mi_test_fvt_tinfo.update_aux_info(
            description=f"Model independent test for {config['CR_fvt_hash']}",
            step=config["step"],
            fvt_scores_tst_SR_test=fvt_scores_tst_SR_test,
            reweights_tst_SR_test=reweights_tst_SR_test,
            reweights_tst_SR_train=reweights_tst_SR_train,
            pi=pi,
        )
    mi_test_fvt_tinfo.save()


@click.command()
@click.option("--config", type=str)
def main(config):
    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    routine(config)


if __name__ == "__main__":
    main()
