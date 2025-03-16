from copy import deepcopy
import datetime
from itertools import product
import pathlib
from typing import Literal
import numpy as np
import pandas as pd
import os

import torch
import tqdm
from dataset import MotherSamples
from fvt_classifier import FvTClassifier
from signal_region import compute_sr_stats, get_SR_CR_cut
from training_info import TrainingInfo
import pickle
from correct_systematic_error import (
    get_histograms,
    just_simple_linear_fit,
)
from events_data import EventsData, get_is_signal
from utils import get_quantiles_with_weights, select_random_true_elements

signal_filename = "HH4b_picoAOD.h5"

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


def correct_systematic_error_new(
    CR_fvt_hash: str,
    nbins_list: list[int],
    bins_stats_type: Literal["fvt", "sr_stats"] = "sr_stats",
    loaded_df: dict[pathlib.Path, pd.DataFrame] = {},
):
    if bins_stats_type == "fvt":
        raise NotImplementedError("FVT bins stats type not implemented")
    if len(loaded_df) == 0:
        path_3b = pathlib.Path("../events/MG3/dataframes/threeTag_picoAOD.h5")
        path_bg4b = pathlib.Path("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
        path_signal = pathlib.Path("../events/MG3/dataframes/HH4b_picoAOD.h5")
        df_3b = pd.read_hdf(path_3b)
        df_bg4b = pd.read_hdf(path_bg4b)
        df_signal = pd.read_hdf(path_signal)
        df_3b["signal"] = False
        df_bg4b["signal"] = False
        df_signal["signal"] = True
        loaded_df = {path_3b: df_3b, path_bg4b: df_bg4b, path_signal: df_signal}

    CR_fvt_tinfo = TrainingInfo.load(CR_fvt_hash)
    SR_stats_hashes = CR_fvt_tinfo.hparams["signal_region"]["SR_stats_hashes"]
    ensemble_mode = CR_fvt_tinfo.hparams["signal_region"]["ensemble_mode"]
    stats_type = CR_fvt_tinfo.hparams["signal_region"]["stats_type"]
    SR_stats_train, SR_stats_tst = compute_sr_stats(
        SR_stats_hashes,
        signal_filename,
        ensemble_mode,
        stats_type,
        loaded_df,
    )
    # base_fvt_train, base_fvt_tst = compute_sr_stats(
    #     SR_stats_hashes,
    #     signal_filename,
    #     ensemble_mode,
    #     "fvt",
    #     loaded_df,
    # )
    ms_idx = TrainingInfo.load(SR_stats_hashes[0]).ms_idx
    msamples = MotherSamples.load(CR_fvt_tinfo.ms_hash)
    train_scdinfo = msamples.scdinfo[ms_idx]
    tst_scdinfo = msamples.scdinfo[~ms_idx]

    df_train = train_scdinfo.fetch_data(loaded_df)
    df_train["signal"] = get_is_signal(train_scdinfo, signal_filename)
    events_train = EventsData.from_dataframe(df_train, features)

    df_tst = tst_scdinfo.fetch_data(loaded_df)
    df_tst["signal"] = get_is_signal(tst_scdinfo, signal_filename)

    SR_cut, _ = get_SR_CR_cut(
        SR_stats_train, events_train, CR_fvt_tinfo.hparams["signal_region"]
    )

    events_train_SR = events_train[SR_stats_train >= SR_cut]
    SR_stats_train_SR = SR_stats_train[SR_stats_train >= SR_cut]
    # base_fvt_train_SR = base_fvt_train[SR_stats_train >= SR_cut]
    SR_idx = SR_stats_tst >= SR_cut
    SR_stats_SR = SR_stats_tst[SR_idx]
    # base_fvt_tst_SR = base_fvt_tst[SR_idx]
    scdinfo_tst_SR = tst_scdinfo[SR_idx]
    events_tst_SR = EventsData.from_dataframe(
        scdinfo_tst_SR.fetch_data(loaded_df), features
    )

    if (
        "fvt_scores_tst_SR" not in CR_fvt_tinfo.aux_info
        or "fvt_scores_train_SR" not in CR_fvt_tinfo.aux_info
    ):
        CR_fvt_model = CR_fvt_tinfo.load_trained_model("best")
        CR_fvt_model.eval()
        CR_fvt_model.to(torch.device("cuda"))
        CR_fvt_model: FvTClassifier

    if "fvt_scores_tst_SR" not in CR_fvt_tinfo.aux_info:
        fvt_scores_tst_SR = (
            CR_fvt_model.predict(events_tst_SR.X_torch)[:, 1].detach().cpu().numpy()
        )
    else:
        fvt_scores_tst_SR = CR_fvt_tinfo.aux_info["fvt_scores_tst_SR"]

    if "fvt_scores_train_SR" not in CR_fvt_tinfo.aux_info:
        fvt_scores_train_SR = (
            CR_fvt_model.predict(events_train_SR.X_torch)[:, 1].detach().cpu().numpy()
        )
    else:
        fvt_scores_train_SR = CR_fvt_tinfo.aux_info["fvt_scores_train_SR"]

    reweights_SR = fvt_scores_tst_SR / (1 - fvt_scores_tst_SR)
    reweights_SR = np.where(events_tst_SR.is_4b, 1, reweights_SR)
    reweights_train_SR = fvt_scores_train_SR / (1 - fvt_scores_train_SR)
    reweights_train_SR = np.where(events_train_SR.is_4b, 1, reweights_train_SR)
    bins_stats_train = SR_stats_train_SR
    bins_stats_tst = SR_stats_SR
    # bins_stats_train = (
    #     SR_stats_train_SR if bins_stats_type == "sr_stats" else base_fvt_train_SR
    # )
    # bins_stats_tst = SR_stats_SR if bins_stats_type == "sr_stats" else base_fvt_tst_SR

    pulls_info_list = []
    for binning_mode, nbins in product(["train", "test"], nbins_list):
        pulls_info = {"binning_mode": binning_mode, "nbins": nbins}
        if binning_mode == "train":
            SR_bins = get_quantiles_with_weights(
                bins_stats_train[events_train_SR.is_3b],
                (reweights_train_SR * events_train_SR.weights)[events_train_SR.is_3b],
                np.linspace(0, 1, nbins + 1),
            )
        else:
            SR_bins = get_quantiles_with_weights(
                bins_stats_tst[events_tst_SR.is_3b],
                (reweights_SR * events_tst_SR.weights)[events_tst_SR.is_3b],
                np.linspace(0, 1, nbins + 1),
            )

        hists = get_histograms(events_tst_SR, bins_stats_tst, SR_bins, reweights_SR)
        if binning_mode == "train":
            V = np.array(hists["3b_rw_sq"] + hists["4b_sq"])
        else:
            V = np.array(hists["4b_sq"]) + np.sum(hists["3b_rw_sq"]) / nbins**2

        y = hists["4b"] - hists["3b_rw"]

        for correction_order in [1, 2]:
            if correction_order == 1:
                X = np.stack([hists["3b_rw"]], axis=1)
                sol, n_eff_bins = just_simple_linear_fit(X, y, V)
                intercept = sol[0]
                slope = 0
            else:
                X = np.stack([hists["3b_rw"], hists["3b_rw_x"]], axis=1)
                sol, n_eff_bins = just_simple_linear_fit(X, y, V)
                intercept = sol[0]
                slope = sol[1]

            pulls_info[f"correction_o{correction_order}"] = {
                "intercept": intercept,
                "slope": slope,
                "n_eff_bins": n_eff_bins,
            }

            corrected_reweights = reweights_SR * (
                1 + intercept + slope * bins_stats_tst
            )
            corrected_weights = (
                np.where(events_tst_SR.is_4b, 1, corrected_reweights)
                * events_tst_SR.weights
            )

            hist_3b_corrected = np.histogram(
                bins_stats_tst[events_tst_SR.is_3b],
                bins=SR_bins,
                weights=corrected_weights[events_tst_SR.is_3b],
            )[0]
            hist_3b_corrected_sq = np.histogram(
                bins_stats_tst[events_tst_SR.is_3b],
                bins=SR_bins,
                weights=corrected_weights[events_tst_SR.is_3b] ** 2,
            )[0]
            hists[f"3b_corrected_o{correction_order}"] = hist_3b_corrected
            hists[f"3b_corrected_o{correction_order}_sq"] = hist_3b_corrected_sq

            if binning_mode == "train":
                hists[f"var_o{correction_order}"] = (
                    hist_3b_corrected_sq + hists["4b_sq"]
                )
            else:
                hists[f"var_o{correction_order}"] = (
                    np.sum(hists["3b_rw_sq"]) / nbins**2 + hists["4b_sq"]
                )

        pulls_info["hists"] = hists
        pulls_info_list.append(pulls_info)

    return pulls_info_list


if __name__ == "__main__":
    nbins_list = [2**i for i in range(2, 11)]
    bins_stats_type = "sr_stats"
    experiment_names = [
        "CR_fvt_training_ensemble_max_smeared",
    ]
    n_3b = 100_0000
    signal_ratios = [0.0, 0.005, 0.0075, 0.01, 0.02]

    TrainingInfo.update_metadata()

    print(
        f"Configs: nbins_list={nbins_list}, bins_stats_type={bins_stats_type}, experiment_names={experiment_names}, n_3b={n_3b}, signal_ratios={signal_ratios}"
    )

    print("Loading dataframes")
    path_3b = pathlib.Path("../events/MG3/dataframes/threeTag_picoAOD.h5")
    path_bg4b = pathlib.Path("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
    path_signal = pathlib.Path("../events/MG3/dataframes/HH4b_picoAOD.h5")
    df_3b = pd.read_hdf(path_3b)
    df_bg4b = pd.read_hdf(path_bg4b)
    df_signal = pd.read_hdf(path_signal)
    df_3b["signal"] = False
    df_bg4b["signal"] = False
    df_signal["signal"] = True
    loaded_df = {path_3b: df_3b, path_bg4b: df_bg4b, path_signal: df_signal}
    print("Dataframes loaded")

    bins_stats_type_str = f"{bins_stats_type}"
    pull_dict_name = f"./data/tmp/pull_by_hashes_{bins_stats_type_str}.pkl"

    if os.path.exists(pull_dict_name):
        with open(pull_dict_name, "rb") as f:
            pull_dict = pickle.load(f)
    else:
        pull_dict = {}

    def process_and_save(hash):
        if hash in pull_dict:
            print(f"[{datetime.datetime.now()}] Hash already processed: {hash}")
            return
        print(f"[{datetime.datetime.now()}] Processing hash: {hash}")
        result = correct_systematic_error_new(
            hash, nbins_list, bins_stats_type, loaded_df
        )
        pull_dict[hash] = result

        with open(pull_dict_name, "wb") as f:
            pickle.dump(pull_dict, f)

    print("Finding hashes to process")
    hashes = TrainingInfo.find(
        {
            "experiment_name": lambda x: x in experiment_names,
            "model": "FvTClassifier",
            "dataset": lambda x: x["n_3b"] == n_3b
            and x["signal_ratio"] in signal_ratios,
            "aux_info_step": 3,
            # "resample": lambda x: x is None or not x,
        }
    )
    # target_hashes = [h for h in hashes if h not in pull_dict.keys()]
    target_hashes = list(hashes)
    # sort by hash names
    target_hashes.sort()

    print(f"Number of hashes to process: {len(target_hashes)}")
    print("Processing hashes starting")

    for hash in tqdm.tqdm(target_hashes):
        process_and_save(hash)

    print("Processing hashes done")
