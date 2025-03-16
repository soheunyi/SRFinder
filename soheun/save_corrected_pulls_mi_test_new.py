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


def correct_systematic_error_new_mi_test(
    mi_test_hash: str,
    nbins_list: list[int],
    loaded_df: dict[pathlib.Path, pd.DataFrame] = {},
):

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

    mi_test_tinfo = TrainingInfo.load(mi_test_hash)
    CR_fvt_hash = mi_test_tinfo.hparams["CR_fvt_hash"]
    CR_fvt_tinfo = TrainingInfo.load(CR_fvt_hash)
    SR_stats_hashes = CR_fvt_tinfo.hparams["signal_region"]["SR_stats_hashes"]
    ensemble_mode = CR_fvt_tinfo.hparams["signal_region"]["ensemble_mode"]
    stats_type = CR_fvt_tinfo.hparams["signal_region"]["stats_type"]
    signal_filename = CR_fvt_tinfo.hparams["dataset"]["signal_filename"]

    ms_idx = TrainingInfo.load(SR_stats_hashes[0]).ms_idx
    msamples = MotherSamples.load(CR_fvt_tinfo.ms_hash)

    train_scdinfo = msamples.scdinfo[ms_idx]
    df_train = train_scdinfo.fetch_data(loaded_df)
    df_train["signal"] = get_is_signal(train_scdinfo, signal_filename)
    events_train = EventsData.from_dataframe(df_train, features)

    tst_scdinfo = msamples.scdinfo[~ms_idx]
    df_tst = tst_scdinfo.fetch_data(loaded_df)
    df_tst["signal"] = get_is_signal(tst_scdinfo, signal_filename)
    events_tst = EventsData.from_dataframe(df_tst, features)

    SR_stats_train, SR_stats_tst = compute_sr_stats(
        SR_stats_hashes, signal_filename, ensemble_mode, stats_type, loaded_df
    )
    SR_cut, _ = get_SR_CR_cut(
        SR_stats_train, events_train, CR_fvt_tinfo.hparams["signal_region"]
    )
    SR_idx = SR_stats_tst >= SR_cut

    SR_3b_idx = events_tst.is_3b & SR_idx
    SR_4b_idx = events_tst.is_4b & SR_idx

    mi_test_test_ratio = mi_test_tinfo.hparams["mi_test_dataset"]["test_ratio"]
    mi_test_data_seed = mi_test_tinfo.hparams["mi_test_dataset"]["data_seed"]

    # select test_ratio of the events in the SR for 3b and 4b
    SR_3b_test_idx = select_random_true_elements(
        SR_3b_idx,
        mi_test_test_ratio,
        mi_test_data_seed,
    )
    SR_4b_test_idx = select_random_true_elements(
        SR_4b_idx,
        mi_test_test_ratio,
        mi_test_data_seed,
    )

    SR_test_idx = SR_3b_test_idx | SR_4b_test_idx
    SR_train_idx = SR_idx & ~SR_test_idx

    events_tst_SR_train = events_tst[SR_train_idx]
    reweights_tst_SR_train = mi_test_tinfo.aux_info["reweights_tst_SR_train"]

    events_tst_SR_test = events_tst[SR_test_idx]
    reweights_tst_SR_test = mi_test_tinfo.aux_info["reweights_tst_SR_test"]

    bins_stats_train = mi_test_tinfo.aux_info["fvt_scores_tst_SR_train"]
    bins_stats_test = mi_test_tinfo.aux_info["fvt_scores_tst_SR_test"]

    pulls_info_list = []
    for binning_mode, nbins in product(["train", "test"], nbins_list):
        pulls_info = {"binning_mode": binning_mode, "nbins": nbins}
        if binning_mode == "train":
            SR_bins = get_quantiles_with_weights(
                bins_stats_train[events_tst_SR_train.is_3b],
                (reweights_tst_SR_train * events_tst_SR_train.weights)[
                    events_tst_SR_train.is_3b
                ],
                np.linspace(0, 1, nbins + 1),
            )
        else:
            SR_bins = get_quantiles_with_weights(
                bins_stats_test[events_tst_SR_test.is_3b],
                (reweights_tst_SR_test * events_tst_SR_test.weights)[
                    events_tst_SR_test.is_3b
                ],
                np.linspace(0, 1, nbins + 1),
            )

        hists = get_histograms(
            events_tst_SR_test, bins_stats_test, SR_bins, reweights_tst_SR_test
        )
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

            corrected_reweights = reweights_tst_SR_test * (
                1 + intercept + slope * bins_stats_test
            )
            corrected_weights = (
                np.where(events_tst_SR_test.is_4b, 1, corrected_reweights)
                * events_tst_SR_test.weights
            )

            hist_3b_corrected = np.histogram(
                bins_stats_test[events_tst_SR_test.is_3b],
                bins=SR_bins,
                weights=corrected_weights[events_tst_SR_test.is_3b],
            )[0]
            hist_3b_corrected_sq = np.histogram(
                bins_stats_test[events_tst_SR_test.is_3b],
                bins=SR_bins,
                weights=corrected_weights[events_tst_SR_test.is_3b] ** 2,
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
    experiment_names = ["mi_test"]
    n_3b = 100_0000
    signal_ratios = [0.0, 0.005, 0.0075, 0.01, 0.02]

    TrainingInfo.update_metadata()

    print(
        f"Configs: nbins_list={nbins_list}, experiment_names={experiment_names}, n_3b={n_3b}, signal_ratios={signal_ratios}"
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

    pull_dict_name = f"./data/tmp/pull_by_hashes_mi_test.pkl"

    if os.path.exists(pull_dict_name):
        with open(pull_dict_name, "rb") as f:
            pull_dict = pickle.load(f)
    else:
        pull_dict = {}

    def process_and_save(hash):
        print(f"[{datetime.datetime.now()}] Processing hash: {hash}")
        existing_nbins = pull_dict.get(hash, {}).get("hists", {}).keys()
        nbins_to_save = [nbins for nbins in nbins_list if nbins not in existing_nbins]
        if len(nbins_to_save) == 0:
            print(f"[{datetime.datetime.now()}] No new nbins to save for hash: {hash}")
            return
        result = correct_systematic_error_new_mi_test(hash, nbins_to_save, loaded_df)
        if hash not in pull_dict:
            pull_dict[hash] = result
        else:
            for key in result.keys():
                pull_dict[hash][key].update(result[key])

        with open(pull_dict_name, "wb") as f:
            pickle.dump(pull_dict, f)

    print("Finding hashes to process")
    hashes = TrainingInfo.find(
        {
            "experiment_name": lambda x: x in experiment_names,
            "model": "FvTClassifier",
            "dataset": lambda x: x["n_3b"] == n_3b
            and x["signal_ratio"] in signal_ratios,
            "aux_info_step": 4,
            "resample": lambda x: x is None or not x,
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
