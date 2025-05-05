from copy import deepcopy
import datetime
from itertools import product
import pathlib
from typing import Literal
import numpy as np
import pandas as pd
import os
import click
import tqdm
import torch
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
    bin_stats_type: Literal["fvt", "smeared"],
    bin_ensemble_mode: Literal["mean", "max"],
    loaded_df: dict[pathlib.Path, pd.DataFrame] = {},
):
    # if bin_stats_type == "fvt":
    #     raise NotImplementedError("FvT bins stats type not implemented")
    if len(loaded_df) == 0:
        path_3b = pathlib.Path("../events/MG3/dataframes/threeTag_picoAOD.h5")
        path_bg4b = pathlib.Path("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
        path_signal = pathlib.Path("../events/MG3/dataframes/HH4b_picoAOD.h5")
        path_signal_HH4b_400 = pathlib.Path("../events/MG3/dataframes/HH4b_400.h5")
        df_3b = pd.read_hdf(path_3b)
        df_bg4b = pd.read_hdf(path_bg4b)
        df_signal = pd.read_hdf(path_signal)
        df_signal_HH4b_400 = pd.read_hdf(path_signal_HH4b_400)
        df_3b["signal"] = False
        df_bg4b["signal"] = False
        df_signal["signal"] = True
        df_signal_HH4b_400["signal"] = True
        loaded_df = {
            path_3b: df_3b,
            path_bg4b: df_bg4b,
            path_signal: df_signal,
            path_signal_HH4b_400: df_signal_HH4b_400,
        }

    CR_fvt_tinfo = TrainingInfo.load(CR_fvt_hash)
    SR_stats_hashes = CR_fvt_tinfo.hparams["signal_region"]["SR_stats_hashes"]
    CR_stats_type = CR_fvt_tinfo.hparams["signal_region"]["stats_type"]
    CR_ensemble_mode = CR_fvt_tinfo.hparams["signal_region"]["ensemble_mode"]

    signal_filename = CR_fvt_tinfo.hparams["dataset"]["signal_filename"]
    SR_stats_train, SR_stats_tst = compute_sr_stats(
        SR_stats_hashes,
        signal_filename,
        CR_ensemble_mode,
        CR_stats_type,
    )
    if CR_stats_type == "fvt" and bin_stats_type == "smeared":
        bin_stats_train, bin_stats_tst = compute_sr_stats(
            SR_stats_hashes,
            signal_filename,
            bin_ensemble_mode,
            "fvt",
        )
    else:
        bin_stats_train, bin_stats_tst = compute_sr_stats(
            SR_stats_hashes,
            signal_filename,
            bin_ensemble_mode,
            bin_stats_type,
        )

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
    bin_stats_train_SR = bin_stats_train[SR_stats_train >= SR_cut]
    SR_idx = SR_stats_tst >= SR_cut
    bin_stats_tst_SR = bin_stats_tst[SR_idx]
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
    if len(reweights_SR) != len(events_tst_SR.is_4b):
        print(
            f"len(reweights_SR) != len(events_tst_SR.is_4b): {len(reweights_SR)} != {len(events_tst_SR.is_4b)}"
        )
        print(f"hash: {CR_fvt_hash}")
        return None
    reweights_SR = np.where(events_tst_SR.is_4b, 1, reweights_SR)
    reweights_train_SR = fvt_scores_train_SR / (1 - fvt_scores_train_SR)
    if len(reweights_train_SR) != len(events_train_SR.is_4b):
        print(
            f"len(reweights_train_SR) != len(events_train_SR.is_4b): {len(reweights_train_SR)} != {len(events_train_SR.is_4b)}"
        )
        print(f"hash: {CR_fvt_hash}")
        return None
    reweights_train_SR = np.where(events_train_SR.is_4b, 1, reweights_train_SR)

    pulls_info_list = []
    for binning_mode, nbins in product(["train", "test"], nbins_list):
        pulls_info = {"binning_mode": binning_mode, "nbins": nbins}
        if binning_mode == "train":
            SR_bins = get_quantiles_with_weights(
                bin_stats_train_SR[events_train_SR.is_3b],
                (reweights_train_SR * events_train_SR.weights)[events_train_SR.is_3b],
                np.linspace(0, 1, nbins + 1),
            )
        else:
            SR_bins = get_quantiles_with_weights(
                bin_stats_tst_SR[events_tst_SR.is_3b],
                (reweights_SR * events_tst_SR.weights)[events_tst_SR.is_3b],
                np.linspace(0, 1, nbins + 1),
            )

        hists = get_histograms(events_tst_SR, bin_stats_tst_SR, SR_bins, reweights_SR)
        if binning_mode == "train":
            V = np.array(hists["3b_rw_sq"] + hists["4b_sq"])
        else:
            V = np.array(hists["4b_sq"]) + np.sum(hists["3b_rw_sq"]) / nbins**2

        y = hists["4b"] - hists["3b_rw"]

        for correction_order in [1, 2, 3]:
            if correction_order == 1:
                X = np.stack([hists["3b_rw"]], axis=1)
                sol, n_eff_bins = just_simple_linear_fit(X, y, V)
                c0 = sol[0]
                c1 = 0
                c2 = 0
            elif correction_order == 2:
                X = np.stack([hists["3b_rw"], hists["3b_rw_x"]], axis=1)
                sol, n_eff_bins = just_simple_linear_fit(X, y, V)
                c0 = sol[0]
                c1 = sol[1]
                c2 = 0
            elif correction_order == 3:
                X = np.stack(
                    [hists["3b_rw"], hists["3b_rw_x"], hists["3b_rw_x_sq"]], axis=1
                )
                sol, n_eff_bins = just_simple_linear_fit(X, y, V)
                c0 = sol[0]
                c1 = sol[1]
                c2 = sol[2]
            else:
                raise ValueError(f"Invalid correction order: {correction_order}")
            pulls_info[f"correction_o{correction_order}"] = {
                "c0": c0,
                "c1": c1,
                "c2": c2,
                "n_eff_bins": n_eff_bins,
            }

            corrected_reweights = reweights_SR * (
                1 + c0 + c1 * bin_stats_tst_SR + c2 * bin_stats_tst_SR**2
            )
            corrected_weights = (
                np.where(events_tst_SR.is_4b, 1, corrected_reweights)
                * events_tst_SR.weights
            )

            hist_3b_corrected = np.histogram(
                bin_stats_tst_SR[events_tst_SR.is_3b],
                bins=SR_bins,
                weights=corrected_weights[events_tst_SR.is_3b],
            )[0]
            hist_3b_corrected_sq = np.histogram(
                bin_stats_tst_SR[events_tst_SR.is_3b],
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


@click.command()
@click.option("--bin_stats_type", type=str, required=True)
@click.option("--bin_ensemble_mode", type=str, required=True)
@click.option("--experiment_name", type=str, required=True)
def save_pulls(
    bin_stats_type: Literal["fvt", "smeared"],
    bin_ensemble_mode: Literal["mean", "max"],
    experiment_name: str,
):
    nbins_list = [2**i for i in range(2, 9)]
    n_3b = 100_0000
    signal_ratios = [0.0, 0.005, 0.0075, 0.01, 0.02]

    # TrainingInfo.update_metadata()

    print(
        f"Configs: nbins_list={nbins_list}, bin_stats_type={bin_stats_type}, bin_ensemble_mode={bin_ensemble_mode}, experiment_name={experiment_name}, n_3b={n_3b}, signal_ratios={signal_ratios}"
    )

    print("Loading dataframes")
    path_3b = pathlib.Path("../events/MG3/dataframes/threeTag_picoAOD.h5")
    path_bg4b = pathlib.Path("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
    path_signal = pathlib.Path("../events/MG3/dataframes/HH4b_picoAOD.h5")
    path_signal_HH4b_800 = pathlib.Path("../events/MG3/dataframes/HH4b_800.h5")
    path_signal_HH4b_400 = pathlib.Path("../events/MG3/dataframes/HH4b_400.h5")
    df_3b = pd.read_hdf(path_3b)
    df_bg4b = pd.read_hdf(path_bg4b)
    df_signal = pd.read_hdf(path_signal)
    df_signal_HH4b_800 = pd.read_hdf(path_signal_HH4b_800)
    df_signal_HH4b_400 = pd.read_hdf(path_signal_HH4b_400)
    df_3b["signal"] = False
    df_bg4b["signal"] = False
    df_signal["signal"] = True
    df_signal_HH4b_800["signal"] = True
    df_signal_HH4b_400["signal"] = True
    loaded_df = {
        path_3b: df_3b,
        path_bg4b: df_bg4b,
        path_signal: df_signal,
        path_signal_HH4b_800: df_signal_HH4b_800,
        path_signal_HH4b_400: df_signal_HH4b_400,
    }

    pull_dict_name = f"./data/pulls/pull_by_hashes_{bin_stats_type}_{bin_ensemble_mode}_{experiment_name}.pkl"

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
            hash, nbins_list, bin_stats_type, bin_ensemble_mode, loaded_df
        )
        if result is None:
            return
        pull_dict[hash] = result

        with open(pull_dict_name, "wb") as f:
            pickle.dump(pull_dict, f)

    print("Finding hashes to process")
    hashes, hparams = TrainingInfo.find(
        {
            "experiment_name": lambda x: x == experiment_name,
            "model": "FvTClassifier",
            "dataset": lambda x: x["n_3b"] == n_3b
            and x["signal_ratio"] in signal_ratios,
            "aux_info_step": 3,
            # "signal_region": lambda x: x["stats_type"] == "fvt",
        },
        return_hparams=True,
    )
    target_hashes = list(hashes)
    # sort by hash names
    target_hashes.sort()

    print(f"Number of hashes to process: {len(target_hashes)}")
    print("Processing hashes starting")

    for hash in tqdm.tqdm(target_hashes):
        process_and_save(hash)

    print("Processing hashes done")


if __name__ == "__main__":
    save_pulls()
