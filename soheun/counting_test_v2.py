from copy import deepcopy
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import click
import yaml


from events_data import events_from_scdinfo
from fvt_classifier import FvTClassifier
from training_info import TrainingInfoV2
from tst_info import TSTInfo


###########################################################################################
###########################################################################################
# 1. Load base fvt model and SR_stats (thus SR and CR) that are trained running counting_test_v1.py
# 2. Initialize CR_fvt model with weights from base fvt model, then train it on CR
# 3. Get the estimated 4b probability from CR_fvt model, then reweight SR
# 4. Perform two sample test
###########################################################################################


def require_keys(config: dict, keys: list):
    for key in keys:
        if key not in config:
            raise ValueError(f"Key {key} is missing in the config")


def routine(config: dict):
    print("Experiment Configuration")
    print(config)

    print("Current Time: ", pd.Timestamp.now())

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

    assert config["experiment_name"] != "counting_test_v1"

    signal_ratio = config["signal_ratio"]
    n_3b = config["n_3b"]
    ratio_4b = config["ratio_4b"]
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

    # Load previous training info
    hparam_filter = {
        "signal_ratio": config["signal_ratio"],
        "n_3b": config["n_3b"],
        "ratio_4b": config["ratio_4b"],
        "signal_filename": config["signal_filename"],
        "seed": config["seed"],
    }
    hashes = TSTInfo.find(hparam_filter)
    assert len(hashes) == 1
    tst_info = TSTInfo.load(hashes[0])

    SR_stats = tst_info.SR_stats
    SR_cut = tst_info.SR_cut
    CR_cut = tst_info.CR_cut
    scdinfo_tst = tst_info.scdinfo_tst
    events_tst = events_from_scdinfo(scdinfo_tst, features, signal_filename)

    SR_idx = SR_stats >= SR_cut
    events_tst_SR = events_tst[SR_idx]

    CR_idx = (SR_stats >= CR_cut) & (SR_stats < SR_cut)
    scdinfo_tst_CR = scdinfo_tst[CR_idx]

    CR_fvt_tinfo = TrainingInfoV2(CR_fvt_hparams, scdinfo_tst_CR)
    print("CR FvT Training Hash: ", CR_fvt_tinfo.hash)

    CR_fvt_train_dset, CR_fvt_val_dset = CR_fvt_tinfo.fetch_train_val_tensor_datasets(
        features, "fourTag", "weight"
    )

    pl.seed_everything(seed)
    np.random.seed(seed)

    # Initialize CR model with weights from base model
    CR_fvt_model = FvTClassifier.load_from_checkpoint(
        f"./data/checkpoints/{tst_info.base_fvt_tinfo_hash}_best.ckpt",
    )
    CR_fvt_model.run_name = CR_fvt_tinfo.hash
    # Train CR model
    CR_fvt_model.fit(
        CR_fvt_train_dset,
        CR_fvt_val_dset,
        batch_size=CR_fvt_hparams["batch_size"],
        max_epochs=CR_fvt_hparams["max_epochs"],
        train_seed=CR_fvt_hparams["train_seed"],
    )

    CR_fvt_model.eval()

    cond_prob_4b_est = (
        CR_fvt_model.predict(events_tst_SR.X_torch)[
            :, 1].detach().cpu().numpy()
    )
    reweights = ratio_4b * cond_prob_4b_est / \
        ((1 - ratio_4b) * (1 - cond_prob_4b_est))
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
        base_fvt_tinfo_hash=tst_info.base_fvt_tinfo_hash,
        CR_fvt_tinfo_hash=CR_fvt_tinfo.hash,
    )
    print("Two Sample Test Hash: ", tst_info.hash)

    CR_fvt_tinfo.save()
    tst_info.save()


@click.command()
@click.option("--config", type=str)
def main(config):
    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    routine(config)


if __name__ == "__main__":
    # load base yml file

    main()
