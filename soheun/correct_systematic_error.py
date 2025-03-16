from copy import deepcopy
import pathlib
import time
from typing import Literal
import numpy as np
import pandas as pd
import torch

from fvt_classifier import FvTClassifier
from attention_classifier import AttentionClassifier
from dataset import MotherSamples, split_scdinfo
from events_data import EventsData, events_from_scdinfo, get_is_signal
from signal_region import get_SR_CR_cut, compute_sr_stats
from training_info import TrainingInfo
from utils import get_quantiles_with_weights, select_random_true_elements

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

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
signal_filename = "HH4b_picoAOD.h5"


def solve_quadratic(A: np.ndarray, b: np.ndarray, x1=None, x2=None):
    """
    Solve min_x x^T A x - 2 b^T x
    """
    assert A.shape == (2, 2)
    assert b.shape == (2, 1)
    assert (x1 is None) or (x2 is None)

    if x1 is None and x2 is None:
        sol = np.linalg.inv(A) @ b
        x1 = sol[0, 0]
        x2 = sol[1, 0]
    elif x2 is not None:
        x1 = (b[0, 0] - A[0, 1] * x2) / A[0, 0]
    elif x1 is not None:
        x2 = (b[1, 0] - A[0, 1] * x1) / A[1, 1]

    return x1, x2


def constrained_linear_fit(
    X: np.ndarray,
    y: np.ndarray,
    V: np.ndarray,
    beta_0_min: float = -np.inf,
    beta_0_max: float = np.inf,
    beta_1_min: float = -np.inf,
    beta_1_max: float = np.inf,
):
    assert len(X) == len(y)
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[1] == 2

    if beta_1_min == beta_1_max == 0:
        X = X[:, 0].reshape(-1, 1)
        XT_V_inv = X.T / V.reshape(-1, 1)
        XT_V_inv[np.isnan(XT_V_inv)] = 0
        A = XT_V_inv @ X
        b = XT_V_inv @ y.reshape(-1, 1)
        beta_0 = (b / A)[0, 0]
        beta_0 = np.clip(beta_0, beta_0_min, beta_0_max)
        return beta_0, 0

    XT_V_inv = X.T / V.reshape(1, -1)
    XT_V_inv[np.isnan(XT_V_inv)] = 0
    A = XT_V_inv @ X
    b = XT_V_inv @ y.reshape(-1, 1)

    A_inv = np.linalg.inv(A)
    sol = (A_inv @ b).flatten()

    if (
        sol[0] >= beta_0_min
        and sol[0] <= beta_0_max
        and sol[1] >= beta_1_min
        and sol[1] <= beta_1_max
    ):
        return sol[0], sol[1]

    possible_sol = []
    for x1 in [beta_0_min, beta_0_max]:
        if np.isinf(x1):
            continue
        sol1, sol2 = solve_quadratic(A, b, x1=x1)
        sol2 = np.clip(sol2, beta_1_min, beta_1_max)
        possible_sol.append(np.array([sol1, sol2]))

    for x2 in [beta_1_min, beta_1_max]:
        if np.isinf(x2):
            continue
        sol1, sol2 = solve_quadratic(A, b, x2=x2)
        sol1 = np.clip(sol1, beta_0_min, beta_0_max)
        possible_sol.append(np.array([sol1, sol2]))

    best_sol = None
    best_val = np.inf
    for sol in possible_sol:
        val = sol.T @ A @ sol - 2 * b.T @ sol
        if val < best_val:
            best_val = val
            best_sol = sol

    return best_sol[0], best_sol[1]


def get_histograms(
    events: EventsData,
    x_values: np.ndarray,
    bins: np.ndarray,
    reweights: np.ndarray,
):
    hist_3b = np.histogram(
        x_values[events.is_3b],
        bins=bins,
        weights=events.weights[events.is_3b],
    )[0]
    hist_4b = np.histogram(
        x_values[events.is_4b],
        bins=bins,
        weights=events.weights[events.is_4b],
    )[0]
    hist_3b_rw = np.histogram(
        x_values[events.is_3b],
        bins=bins,
        weights=(events.weights * reweights)[events.is_3b],
    )[0]
    hist_3b_rw_x = np.histogram(
        x_values[events.is_3b],
        bins=bins,
        weights=(events.weights * x_values * reweights)[events.is_3b],
    )[0]
    hist_3b_rw_sq = np.histogram(
        x_values[events.is_3b],
        bins=bins,
        weights=(events.weights**2 * reweights**2)[events.is_3b],
    )[0]
    hist_4b_sq = np.histogram(
        x_values[events.is_4b],
        bins=bins,
        weights=(events.weights**2)[events.is_4b],
    )[0]
    hist_signal = np.histogram(
        x_values[events.is_signal],
        bins=bins,
        weights=events.weights[events.is_signal],
    )[0]
    hist_signal_sq = np.histogram(
        x_values[events.is_signal],
        bins=bins,
        weights=(events.weights[events.is_signal] ** 2),
    )[0]
    hist_bg4b = np.histogram(
        x_values[events.is_bg4b],
        bins=bins,
        weights=events.weights[events.is_bg4b],
    )[0]
    hist_bg4b_sq = np.histogram(
        x_values[events.is_bg4b],
        bins=bins,
        weights=(events.weights[events.is_bg4b] ** 2),
    )[0]
    return {
        "3b": hist_3b,
        "3b_rw": hist_3b_rw,
        "3b_rw_x": hist_3b_rw_x,
        "3b_rw_sq": hist_3b_rw_sq,
        "4b": hist_4b,
        "4b_sq": hist_4b_sq,
        "signal": hist_signal,
        "signal_sq": hist_signal_sq,
        "bg4b": hist_bg4b,
        "bg4b_sq": hist_bg4b_sq,
    }


def correct_systematic_error(
    CR_fvt_hash: str,
    nbins_list: list[int],
    bins_mode: str = "quantile",
    bins_stats_type: Literal["fvt", "sr_stats"] = "fvt",
    intercept_min: float = -np.inf,
    intercept_max: float = np.inf,
    slope_min: float = 0.0,
    slope_max: float = 0.0,
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

    corrections = {nbins: [] for nbins in nbins_list}
    hist_corrected_dict = {}

    CR_fvt_tinfo = TrainingInfo.load(CR_fvt_hash)
    SR_stats_hashes = CR_fvt_tinfo.hparams["signal_region"]["SR_stats_hashes"]
    ensemble_mode = CR_fvt_tinfo.hparams["signal_region"]["ensemble_mode"]
    stats_type = CR_fvt_tinfo.hparams["signal_region"]["stats_type"]
    SR_stats_train, SR_stats_tst = compute_sr_stats(
        SR_stats_hashes, signal_filename, ensemble_mode, stats_type
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
    SR_stats_train_SR = SR_stats_train[SR_stats_train >= SR_cut]
    # logger.info(f"Computed SR stats for {CR_fvt_hash}")
    SR_idx = SR_stats_tst >= SR_cut
    SR_stats_SR = SR_stats_tst[SR_idx]
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
    # logger.info(f"Computed reweights for {CR_fvt_hash}")
    reweights_train_SR = fvt_scores_train_SR / (1 - fvt_scores_train_SR)
    reweights_train_SR = np.where(events_train_SR.is_4b, 1, reweights_train_SR)
    # logger.info(f"Computed reweights for {CR_fvt_hash}")
    bins_stats_train = (
        SR_stats_train_SR if bins_stats_type == "sr_stats" else fvt_scores_train_SR
    )
    bins_stats_tst = SR_stats_SR if bins_stats_type == "sr_stats" else fvt_scores_tst_SR
    for nbins in nbins_list:
        if bins_mode == "quantile":
            SR_bins = get_quantiles_with_weights(
                bins_stats_train[events_train_SR.is_3b],
                (reweights_train_SR * events_train_SR.weights)[events_train_SR.is_3b],
                np.linspace(0, 1, nbins + 1),
            )
        elif bins_mode == "uniform":
            SR_bins = np.linspace(
                np.min(bins_stats_train), np.max(bins_stats_train), nbins + 1
            )

        hists = get_histograms(events_tst_SR, bins_stats_tst, SR_bins, reweights_SR)

        y = hists["4b"] - hists["3b_rw"]
        X = np.stack([hists["3b_rw"], hists["3b_rw_x"]], axis=1)
        V = np.array(hists["3b_sq"] + hists["4b_sq"])

        intercept, slope = constrained_linear_fit(
            X,
            y,
            V,
            beta_0_min=intercept_min,
            beta_0_max=intercept_max,
            beta_1_min=slope_min if nbins > 1 else 0,
            beta_1_max=slope_max if nbins > 1 else 0,
        )
        corrections[nbins].append((intercept, slope))

        corrected_reweights = reweights_SR * (1 + intercept + slope * bins_stats_tst)
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
        hists["3b_corrected"] = hist_3b_corrected
        hists["3b_corrected_sq"] = hist_3b_corrected_sq
        hist_corrected_dict[nbins] = deepcopy(hists)

    return corrections, hist_corrected_dict


def just_simple_linear_fit(
    X: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
):
    assert y.ndim == v.ndim == 1
    assert X.ndim == 2
    assert len(X) == len(y) == len(v)
    # remove 0 values from V
    v_nonzero_idx = np.where(v != 0)[0]
    X_nonzero = X[v_nonzero_idx]
    y_nonzero = y[v_nonzero_idx].reshape(-1, 1)
    v_nonzero = v[v_nonzero_idx]
    n_eff_bins = len(v_nonzero)

    V_inv = np.diag(1 / v_nonzero)
    sol = (
        np.linalg.inv(X_nonzero.T @ V_inv @ X_nonzero) @ X_nonzero.T @ V_inv @ y_nonzero
    )
    return sol.flatten(), n_eff_bins


def correct_systematic_error_mi_test(
    mi_test_hash: str,
    nbins_list: list[int],
    bins_mode: str = "quantile",
    correction_order: int = 1,
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

    corrections = {nbins: [] for nbins in nbins_list}
    hist_corrected_dict = {}

    mi_test_tinfo = TrainingInfo.load(mi_test_hash)

    for key in [
        "fvt_scores_tst_SR_train",
        "fvt_scores_tst_SR_test",
        "reweights_tst_SR_train",
        "reweights_tst_SR_test",
    ]:
        if key not in mi_test_tinfo.aux_info:
            print(f"Key {key} not found in aux_info of {mi_test_hash}")
            continue

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

    mi_fvt_scores_tst_SR_train, mi_fvt_scores_tst_SR_test = (
        mi_test_tinfo.aux_info["fvt_scores_tst_SR_train"],
        mi_test_tinfo.aux_info["fvt_scores_tst_SR_test"],
    )

    bins_stats_train, bins_stats_test = (
        mi_fvt_scores_tst_SR_train,
        mi_fvt_scores_tst_SR_test,
    )

    for nbins in nbins_list:
        if bins_mode == "quantile":
            SR_bins = get_quantiles_with_weights(
                bins_stats_train[events_tst_SR_train.is_3b],
                (reweights_tst_SR_train * events_tst_SR_train.weights)[
                    events_tst_SR_train.is_3b
                ],
                np.linspace(0, 1, nbins + 1),
            )
        elif bins_mode == "uniform":
            SR_bins = np.linspace(
                np.min(bins_stats_train),
                np.max(bins_stats_train),
                nbins + 1,
            )
        SR_bins[0] = np.min([SR_bins[0], np.min(bins_stats_test)])
        SR_bins[-1] = np.max([SR_bins[-1], np.max(bins_stats_test)])

        hists = get_histograms(
            events_tst_SR_test,
            bins_stats_test,
            SR_bins,
            reweights_tst_SR_test,
        )

        y = hists["4b"] - hists["3b_rw"]
        V = np.array(hists["3b_sq"] + hists["4b_sq"])

        if correction_order == 1:
            X = np.stack([hists["3b_rw"]], axis=1)
            sol, n_eff_bins = just_simple_linear_fit(X, y, V)
            intercept = sol[0]
            slope = 0
        elif correction_order == 2:
            X = np.stack([hists["3b_rw"], hists["3b_rw_x"]], axis=1)
            sol, n_eff_bins = just_simple_linear_fit(X, y, V)
            intercept = sol[0]
            slope = sol[1]
        else:
            raise ValueError(f"Invalid correction order: {correction_order}")

        corrections[nbins].append((intercept, slope, n_eff_bins))

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
        hists["3b_corrected"] = hist_3b_corrected
        hists["3b_corrected_sq"] = hist_3b_corrected_sq
        hist_corrected_dict[nbins] = deepcopy(hists)

    return corrections, hist_corrected_dict
