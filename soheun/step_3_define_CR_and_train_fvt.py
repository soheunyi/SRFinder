from copy import deepcopy
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
from training_info import TrainingInfo
from utils import require_keys
from signal_region import get_DRs, get_SR_CR_cut


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
            "signal_region",
            "CR_fvt",
            "previous_step_experiment_name",
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
        config["signal_region"],
        ["4b_in_SR", "4b_in_CR"],
    )
    require_keys(
        config["CR_fvt"],
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
    require_keys(config["CR_fvt"]["optimizer"], ["type", "lr"])
    require_keys(
        config["CR_fvt"]["lr_scheduler"],
        ["type"],
    )
    if config["CR_fvt"]["lr_scheduler"]["type"] == "ReduceLROnPlateau":
        require_keys(
            config["CR_fvt"]["lr_scheduler"],
            ["factor", "threshold", "patience", "cooldown", "min_lr"],
        )
    if config["CR_fvt"]["optimizer"]["type"] == "AdamWScheduleFree":
        require_keys(
            config["CR_fvt"]["optimizer"],
            ["warmup_steps"],
        )
    require_keys(
        config["CR_fvt"]["dataloader"],
        ["batch_size", "batch_size_multiplier", "batch_size_milestones"],
    )

    if config["CR_fvt"]["model"] == "FvTClassifier":
        require_keys(
            config["CR_fvt"],
            ["dim_dijet_features", "dim_quadjet_features"],
        )
        assert isinstance(config["CR_fvt"]["depth"], dict), "depth must be a dictionary"
        require_keys(
            config["CR_fvt"]["depth"],
            ["encoder", "decoder"],
        )
    elif config["CR_fvt"]["model"] == "AttentionClassifier":
        require_keys(
            config["CR_fvt"],
            ["dim_q"],
        )

    signal_ratio = config["dataset"]["signal_ratio"]
    n_3b = config["dataset"]["n_3b"]
    ratio_4b = config["dataset"]["ratio_4b"]
    signal_filename = config["dataset"]["signal_filename"]
    seed = config["dataset"]["seed"]

    CR_fvt_hparams = deepcopy(config["CR_fvt"])
    CR_fvt_hparams["experiment_name"] = config["experiment_name"]
    CR_fvt_hparams["dataset"] = config["dataset"]
    CR_fvt_hparams["signal_region"] = config["signal_region"]

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

    # 1. Find and load encoder & smeared FvT model
    hashes = TrainingInfo.find(
        {
            "dataset": lambda x: (
                x["n_3b"] == n_3b
                and x["ratio_4b"] == ratio_4b
                and x["signal_ratio"] == signal_ratio
                and x["signal_filename"] == signal_filename
                and x["seed"] == seed
            ),
            "aux_info_step": 2,
            "experiment_name": config["previous_step_experiment_name"],
        }
    )
    assert (
        len(hashes) == 1
    ), f"Number of training info must be one, there are {len(hashes)}"
    smeared_fvt_tinfo = TrainingInfo.load(hashes[0])
    CR_fvt_hparams["smeared_fvt_hash"] = smeared_fvt_tinfo.hash
    base_encoder_hash = smeared_fvt_tinfo.hparams["encoder_hash"]
    CR_fvt_hparams["encoder_hash"] = base_encoder_hash

    base_fvt_model = TrainingInfo.load(base_encoder_hash).load_trained_model(
        # CR_fvt_hparams["encoder_mode"]
        "best"
    )
    base_fvt_model: FvTClassifier
    base_fvt_model.eval()
    base_fvt_model.to(torch.device("cuda"))

    smeared_fvt_model = smeared_fvt_tinfo.load_trained_model(
        # CR_fvt_hparams["encoder_mode"]
        "best"
    )
    smeared_fvt_model: AttentionClassifier
    smeared_fvt_model.eval()
    smeared_fvt_model.to(torch.device("cuda"))

    # Use the same mother samples and exclude ones used for training base & smeared FvT model
    msamples = MotherSamples.load(smeared_fvt_tinfo.ms_hash)
    events_train = events_from_scdinfo(
        msamples.scdinfo[smeared_fvt_tinfo.ms_idx], features, signal_filename
    )
    events_tst = events_from_scdinfo(
        msamples.scdinfo[~smeared_fvt_tinfo.ms_idx], features, signal_filename
    )

    if ("SR_stats_train" not in smeared_fvt_tinfo.aux_info) or (
        "SR_stats_tst" not in smeared_fvt_tinfo.aux_info
    ):
        print("Calculating SR_stats_train and SR_stats_tst")
        events_all = EventsData.merge([events_train, events_tst])
        gamma_base_all, gamma_smeared_all, _ = get_DRs(
            events_all, base_fvt_model, smeared_fvt_model, return_repr=True
        )
        SR_stats_all = np.log(gamma_base_all / gamma_smeared_all)

        SR_stats_train = SR_stats_all[: len(events_train)]
        SR_stats_tst = SR_stats_all[len(events_train) :]

        smeared_fvt_tinfo.update_aux_info(
            SR_stats_train=SR_stats_train,
            SR_stats_tst=SR_stats_tst,
        )
        smeared_fvt_tinfo.save()
    else:
        print("Using cached SR_stats_train and SR_stats_tst")
        SR_stats_train = smeared_fvt_tinfo.aux_info["SR_stats_train"]
        SR_stats_tst = smeared_fvt_tinfo.aux_info["SR_stats_tst"]

    SR_cut, CR_cut = get_SR_CR_cut(
        SR_stats_train, events_train, config["signal_region"]
    )
    CR_idx = (SR_stats_tst >= CR_cut) & (SR_stats_tst < SR_cut)

    tst_ms_idx = ~smeared_fvt_tinfo.ms_idx
    tst_ms_idx_int = tst_ms_idx.nonzero()[0]
    CR_ms_idx_int = tst_ms_idx_int[CR_idx]
    CR_ms_idx_bool = np.zeros_like(tst_ms_idx, dtype=bool)
    CR_ms_idx_bool[CR_ms_idx_int] = True

    CR_fvt_tinfo = TrainingInfo(
        CR_fvt_hparams,
        ms_hash=smeared_fvt_tinfo.ms_hash,
        ms_idx=CR_ms_idx_bool,
    )

    CR_fvt_train_dset, CR_fvt_val_dset = CR_fvt_tinfo.fetch_train_val_tensor_datasets(
        features,
        label="fourTag",
        weight="weight",
        label_dtype=torch.long,
    )

    pl.seed_everything(CR_fvt_hparams["model_seed"])

    if CR_fvt_hparams["model"] == "AttentionClassifier":
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
    )

    CR_fvt_tinfo.update_aux_info(
        description=f"FvT on Control Region, based on base FvT={base_encoder_hash}, smeared FvT={smeared_fvt_tinfo.hash}",
        step=3,
    )
    CR_fvt_tinfo.save()
    TrainingInfo.update_metadata()


@click.command()
@click.option("--config", type=str)
def main(config):
    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    routine(config)


if __name__ == "__main__":
    main()
