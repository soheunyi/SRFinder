from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import tqdm

from fvt_classifier import FvTClassifier
from attention_classifier import AttentionClassifier
from dataset import MotherSamples, split_scdinfo
from events_data import EventsData, events_from_scdinfo, get_is_signal
from signal_region import get_SR_CR_cut
from training_info import TrainingInfo


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
        A = X.T @ np.linalg.inv(V) @ X
        b = X.T @ np.linalg.inv(V) @ y.reshape(-1, 1)
        beta_0 = (b / A)[0, 0]
        beta_0 = np.clip(beta_0, beta_0_min, beta_0_max)
        return beta_0, 0

    A = X.T @ np.linalg.inv(V) @ X
    b = X.T @ np.linalg.inv(V) @ y.reshape(-1, 1)

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
    hist_3b_x = np.histogram(
        x_values[events.is_3b],
        bins=bins,
        weights=(events.weights * x_values)[events.is_3b],
    )[0]
    hist_3b_sq = np.histogram(
        x_values[events.is_3b],
        bins=bins,
        weights=(events.weights**2 * reweights**2)[events.is_3b],
    )[0]
    hist_4b_sq = np.histogram(
        x_values[events.is_4b],
        bins=bins,
        weights=(events.weights**2)[events.is_4b],
    )[0]
    return {
        "3b": hist_3b,
        "3b_rw": hist_3b_rw,
        "3b_x": hist_3b_x,
        "3b_sq": hist_3b_sq,
        "4b": hist_4b,
        "4b_sq": hist_4b_sq,
    }


def correct_systematic_error(
    CR_fvt_hash: str,
    nbins_list: list[int],
    bins_mode: str = "quantile",
    correction_split_seed: int = 42,
    correction_split_ratio: float = 0.2,
    intercept_min: float = -np.inf,
    intercept_max: float = np.inf,
    slope_min: float = -np.inf,
    slope_max: float = np.inf,
    raw_df_list: list[pd.DataFrame] = [],
):
    if len(raw_df_list) == 0:
        df_3b = pd.read_hdf("../events/MG3/dataframes/threeTag_picoAOD.h5")
        df_bg4b = pd.read_hdf("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
        df_signal = pd.read_hdf("../events/MG3/dataframes/HH4b_picoAOD.h5")
        df_3b["signal"] = False
        df_bg4b["signal"] = False
        df_signal["signal"] = True
        raw_df_list = [df_3b, df_bg4b, df_signal]

    corrections = {nbins: [] for nbins in nbins_list}
    hist_corrected_dict = {}

    CR_fvt_tinfo = TrainingInfo.load(CR_fvt_hash)
    base_encoder_hash = CR_fvt_tinfo.hparams["encoder_hash"]
    base_fvt_tinfo = TrainingInfo.load(base_encoder_hash)
    smeared_fvt_hash = CR_fvt_tinfo.hparams["smeared_fvt_hash"]
    smeared_fvt_tinfo = TrainingInfo.load(smeared_fvt_hash)

    base_fvt_model = base_fvt_tinfo.load_trained_model(
        CR_fvt_tinfo.hparams["encoder_mode"]
    )
    base_fvt_model.eval()
    base_fvt_model.to(torch.device("cuda"))
    base_fvt_model: FvTClassifier

    smeared_fvt_model = smeared_fvt_tinfo.load_trained_model(
        CR_fvt_tinfo.hparams["encoder_mode"]
    )
    smeared_fvt_model.eval()
    smeared_fvt_model.to(torch.device("cuda"))
    smeared_fvt_model: AttentionClassifier

    # Use the same mother samples and exclude ones used for training base & smeared FvT model
    msamples = MotherSamples.load(smeared_fvt_tinfo.ms_hash)
    tst_scdinfo = msamples.scdinfo[~smeared_fvt_tinfo.ms_idx]
    train_scdinfo = msamples.scdinfo[smeared_fvt_tinfo.ms_idx]

    df_train = train_scdinfo.fetch_data_with_loaded_df(raw_df_list)
    df_train["signal"] = get_is_signal(train_scdinfo, signal_filename)
    events_train = EventsData.from_dataframe(df_train, features)

    df_tst = tst_scdinfo.fetch_data_with_loaded_df(raw_df_list)
    df_tst["signal"] = get_is_signal(tst_scdinfo, signal_filename)

    SR_stats_train = smeared_fvt_tinfo.aux_info["SR_stats_train"]
    SR_stats_tst = smeared_fvt_tinfo.aux_info["SR_stats_tst"]
    SR_cut, _ = get_SR_CR_cut(
        SR_stats_train, events_train, CR_fvt_tinfo.hparams["signal_region"]
    )
    SR_idx = SR_stats_tst >= SR_cut
    SR_stats_SR = SR_stats_tst[SR_idx]
    scdinfo_SR = tst_scdinfo[SR_idx]
    events_SR = EventsData.from_dataframe(
        scdinfo_SR.fetch_data_with_loaded_df(raw_df_list), features
    )

    CR_fvt_model = CR_fvt_tinfo.load_trained_model("best")
    CR_fvt_model.eval()
    CR_fvt_model.to(torch.device("cuda"))
    CR_fvt_model: FvTClassifier
    fvt_scores_SR = CR_fvt_model.predict(events_SR.X_torch)[:, 1].detach().cpu().numpy()
    reweights_SR = fvt_scores_SR / (1 - fvt_scores_SR)
    reweights_SR = np.where(events_SR.is_4b, 1, reweights_SR)

    np.random.seed(correction_split_seed)
    train_len = int(len(scdinfo_SR) * correction_split_ratio)
    train_idx = np.array([True] * train_len + [False] * (len(scdinfo_SR) - train_len))
    np.random.shuffle(train_idx)

    for nbins in nbins_list:
        if bins_mode == "quantile":
            SR_bins = np.quantile(SR_stats_SR, np.linspace(0, 1, nbins + 1))
        elif bins_mode == "uniform":
            SR_bins = np.linspace(np.min(SR_stats_SR), np.max(SR_stats_SR), nbins + 1)

        hists_train = get_histograms(
            events_SR[train_idx],
            SR_stats_SR[train_idx],
            SR_bins,
            reweights_SR[train_idx],
        )
        hists_test = get_histograms(
            events_SR[~train_idx],
            SR_stats_SR[~train_idx],
            SR_bins,
            reweights_SR[~train_idx],
        )
        y = hists_train["4b"] - hists_train["3b_rw"]
        X = np.stack([hists_train["3b"], hists_train["3b_x"]], axis=1)
        V = np.diag(hists_train["3b_sq"] + hists_test["4b_sq"])
        # V = np.diag(hists_train["4b"])

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
        SR_stats_SR_test = SR_stats_SR[~train_idx]
        test_is_3b = events_SR[~train_idx].is_3b

        corrected_reweights = (
            reweights_SR[~train_idx] + intercept + slope * SR_stats_SR_test
        )
        corrected_weights = (
            np.where(events_SR[~train_idx].is_4b, 1, corrected_reweights)
            * events_SR[~train_idx].weights
        )

        test_is_3b = events_SR[~train_idx].is_3b
        hist_3b_corrected = np.histogram(
            SR_stats_SR_test[test_is_3b],
            bins=SR_bins,
            weights=corrected_weights[test_is_3b],
        )[0]
        hist_3b_corrected_sq = np.histogram(
            SR_stats_SR_test[test_is_3b],
            bins=SR_bins,
            weights=corrected_weights[test_is_3b] ** 2,
        )[0]
        hists_test["3b_corrected"] = hist_3b_corrected
        hists_test["3b_corrected_sq"] = hist_3b_corrected_sq
        hist_corrected_dict[nbins] = deepcopy(hists_test)

    return corrections, hist_corrected_dict
