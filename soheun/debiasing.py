import os
import sys
from typing import Callable
import numpy as np
import pandas as pd
import torch
import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from events_data import events_from_scdinfo, EventsData
from fvt_classifier import FvTClassifier
from plots import hist_events_by_labels
from tst_info import TSTInfo


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


def get_histograms(
    events_original: EventsData, values: np.ndarray, bins, reweights: np.ndarray = 1
):
    is_3b = events_original.is_3b
    is_bg4b = events_original.is_bg4b
    is_signal = events_original.is_signal

    hist_3b, _ = np.histogram(
        values[is_3b], bins=bins, weights=events_original.weights[is_3b]
    )
    hist_bg4b, _ = np.histogram(
        values[is_bg4b], bins=bins, weights=events_original.weights[is_bg4b]
    )
    hist_signal, _ = np.histogram(
        values[is_signal], bins=bins, weights=events_original.weights[is_signal]
    )
    hist_3b_rw, _ = np.histogram(
        values[is_3b], bins=bins, weights=(events_original.weights * reweights)[is_3b]
    )
    hist_3b_rw_sq, _ = np.histogram(
        values[is_3b],
        bins=bins,
        weights=(events_original.weights * reweights**2)[is_3b],
    )

    hist_4b = hist_bg4b + hist_signal
    hist_all = hist_3b + hist_4b

    return {
        "3b": hist_3b,
        "bg4b": hist_bg4b,
        "signal": hist_signal,
        "4b": hist_4b,
        "3b_rw": hist_3b_rw,
        "3b_rw_sq": hist_3b_rw_sq,
        "all": hist_all,
    }


def get_bias_fn(
    events: EventsData, probs_4b_est: np.ndarray, calibration_nbin: int
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get the bias function for the debiasing.
    events should not be given with reweighted weights.

    Args:
        events: EventsData,
        probs_4b_est: np.ndarray,
        calibration_nbin: int,

    Returns:
        bias_fn: function, the bias function for the debiasing.
    """
    assert len(events) == len(probs_4b_est)

    calibration_bins = np.linspace(
        np.min(probs_4b_est), np.max(probs_4b_est), calibration_nbin + 1
    )
    midpoints = (calibration_bins[1:] + calibration_bins[:-1]) / 2
    histograms = get_histograms(events, probs_4b_est, calibration_bins, 1)
    hist_4b = histograms["4b"]
    hist_all = histograms["all"]
    hist_all = np.where(hist_all > 0, hist_all, 1)  # avoid division by zero
    mean_probs_4b = np.where(hist_all > 0, hist_4b / hist_all, midpoints)
    calibration_error = midpoints - mean_probs_4b

    def piecewise_linear(x_knots, y_knots):
        def piecewise_linear_inner(x):
            return np.interp(x, x_knots, y_knots)

        return piecewise_linear_inner

    calib_error_fn = piecewise_linear(midpoints, calibration_error)

    def bias_fn(x):
        return calib_error_fn(x) * (1 / (1 - x)) ** 2

    return bias_fn


def get_histogram_info(events_original: EventsData, values, bins, reweights, bias=None):
    assert len(events_original) == len(values)

    if isinstance(bins, int):
        # calculate quantiles
        q = np.linspace(0, 1, bins + 1)
        bins = np.quantile(values, q)

    if bias is not None:
        hist_bias, _ = np.histogram(
            values[events_original.is_3b],
            bins=bins,
            weights=(events_original.weights * bias)[events_original.is_3b],
        )

    histograms = get_histograms(events_original, values, bins, reweights)
    hist_bg4b = histograms["bg4b"]
    hist_4b = histograms["4b"]
    hist_3b_rw = histograms["3b_rw"]
    hist_3b_rw_sq = histograms["3b_rw_sq"]

    std_est = np.sqrt(hist_4b + hist_3b_rw_sq)
    is_sampled = std_est > 0

    if bias is not None:
        sigma = (hist_4b - (hist_3b_rw - hist_bias))[is_sampled] / std_est[is_sampled]
    else:
        sigma = (hist_4b - hist_3b_rw)[is_sampled] / std_est[is_sampled]

    sigma_avg = np.sqrt(np.mean(sigma**2))

    if bias is not None:
        sigma_bg4b = (hist_bg4b - (hist_3b_rw - hist_bias))[is_sampled] / std_est[
            is_sampled
        ]
    else:
        sigma_bg4b = (hist_bg4b - hist_3b_rw)[is_sampled] / std_est[is_sampled]

    sigma_avg_bg4b = np.sqrt(np.mean(sigma_bg4b**2))
    df = np.sum(is_sampled)

    histograms.update(
        {
            "bias": hist_bias,
            "std_est": std_est,
            "sigma": sigma,
            "sigma_avg": sigma_avg,
            "sigma_bg4b": sigma_bg4b,
            "sigma_avg_bg4b": sigma_avg_bg4b,
            "df": df,
        }
    )

    return histograms


def get_bias_binwise(
    events: EventsData,
    values: np.ndarray,
    nbins: int,
    probs_4b_est: np.ndarray,
    calibration_nbin: int,
    cheating=False,
):
    """
    Get the bias function for the debiasing.
    events should not be given with reweighted weights.

    Args:
        events: EventsData,
        probs_4b_est: np.ndarray,
        calibration_nbin: int,
        nbins: int,
    """
    q = np.linspace(0, 1, nbins + 1)
    bins = np.quantile(values, q)
    bin_idx = np.digitize(values, bins) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    tst_idx = []
    bias = []

    for b in range(nbins):
        members = np.where(bin_idx == b)[0]
        events_bin = events[members]
        probs_4b_est_bin = probs_4b_est[members]
        # randomly sample 1/3 of the events
        if cheating:
            idx_bias_fn = np.random.choice(
                len(events_bin), size=len(events_bin), replace=False
            )
        else:
            idx_bias_fn = np.random.choice(
                len(events_bin), size=len(events_bin) // 2, replace=False
            )
        events_bin_bias_fn = events_bin[idx_bias_fn]
        probs_4b_est_bin_bias_fn = probs_4b_est_bin[idx_bias_fn]

        bias_fn = get_bias_fn(
            events_bin_bias_fn,
            probs_4b_est_bin_bias_fn,
            calibration_nbin,
        )

        # calculate bias for other 2/3
        if cheating:
            idx_other = idx_bias_fn
        else:
            idx_other = np.setdiff1d(np.arange(len(events_bin)), idx_bias_fn)
        bias_other = bias_fn(probs_4b_est_bin[idx_other])
        bias.extend(bias_other)
        tst_idx.extend(members[idx_other])

    bias = np.array(bias)

    return tst_idx, bias


if __name__ == "__main__":
    plt.rcParams["lines.markersize"] = 3

    verbose = False
    show_plots = False
    n_3b = 100_0000
    device = torch.device("cuda")
    do_tqdm = True
    calibration_nbin = 10
    nbins_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cheating = True
    experiment_name = "counting_test_v2"
    hparam_filter = {
        "experiment_name": lambda x: x in [experiment_name],
        "n_3b": n_3b,
    }

    binwise_debiasing = True
    df_name = f"./data/tsv/tst_results_summary_{experiment_name}_n_3b={n_3b}_debiased_binwise={binwise_debiasing}_cheating={cheating}.tsv"

    if not os.path.exists(df_name):
        tst_results_summary_df = pd.DataFrame()
    else:
        tst_results_summary_df = pd.read_csv(df_name, sep="\t")

    hashes = TSTInfo.find(hparam_filter, sort_by=["signal_ratio", "seed"])
    tst_results_summary = []

    for tstinfo_hash in (pbar := tqdm.tqdm(hashes)):

        tstinfo = TSTInfo.load(tstinfo_hash)
        signal_filename = tstinfo.hparams["signal_filename"]
        seed = tstinfo.hparams["seed"]
        signal_ratio = tstinfo.hparams["signal_ratio"]
        experiment_name = tstinfo.hparams["experiment_name"]

        if tst_results_summary_df.shape[0] > 0:
            if (
                (tst_results_summary_df["signal_ratio"] == signal_ratio)
                & (tst_results_summary_df["seed"] == seed)
            ).any():
                continue

        initialize_with_fvt = True

        scdinfo_tst = tstinfo.scdinfo_tst
        events_tst = events_from_scdinfo(scdinfo_tst, features, signal_filename)

        base_fvt_hash = tstinfo.base_fvt_tinfo_hash
        fvt_model = FvTClassifier.load_from_checkpoint(
            f"./data/checkpoints/{base_fvt_hash}_best.ckpt"
        )
        fvt_model.to(device)
        fvt_model.freeze()

        CR_fvt_hash = tstinfo.CR_fvt_tinfo_hash
        CR_model = FvTClassifier.load_from_checkpoint(
            f"./data/checkpoints/{CR_fvt_hash}_best.ckpt"
        )
        CR_model.to(device)
        CR_model.freeze()

        SR_stats = tstinfo.SR_stats
        SR_cut = tstinfo.SR_cut
        CR_cut = tstinfo.CR_cut
        in_SR = SR_stats >= SR_cut
        in_CR = (SR_stats < SR_cut) & (SR_stats >= CR_cut)

        ratio_4b = tstinfo.hparams["ratio_4b"]

        for rw_method in ["CR", "base"]:
            if rw_method == "CR":
                probs_4b_est = (
                    CR_model.predict(events_tst.X_torch, do_tqdm=do_tqdm)
                    .detach()
                    .cpu()
                    .numpy()[:, 1]
                )
            else:
                probs_4b_est = (
                    fvt_model.predict(events_tst.X_torch, do_tqdm=do_tqdm)
                    .detach()
                    .cpu()
                    .numpy()[:, 1]
                )

            reweights = ratio_4b * probs_4b_est / ((1 - ratio_4b) * (1 - probs_4b_est))

            events_SR = events_tst[in_SR]
            SR_stats_SR = SR_stats[in_SR]
            probs_4b_est_SR = probs_4b_est[in_SR]

            events_CR = events_tst[in_CR]
            SR_stats_CR = SR_stats[in_CR]
            probs_4b_est_CR = probs_4b_est[in_CR]

            for ax_cnt, nbins in enumerate(nbins_list):
                tst_idx_SR, bias_3b_rw_SR = get_bias_binwise(
                    events_SR,
                    SR_stats_SR,
                    nbins,
                    probs_4b_est_SR,
                    calibration_nbin,
                    cheating=cheating,
                )
                tst_idx_CR, bias_3b_rw_CR = get_bias_binwise(
                    events_CR,
                    SR_stats_CR,
                    nbins,
                    probs_4b_est_CR,
                    calibration_nbin,
                    cheating=cheating,
                )

                hist_info_SR = get_histogram_info(
                    events_SR[tst_idx_SR],
                    SR_stats_SR[tst_idx_SR],
                    nbins,
                    reweights=reweights[in_SR][tst_idx_SR],
                    bias=bias_3b_rw_SR,
                )
                hist_info_CR = get_histogram_info(
                    events_CR[tst_idx_CR],
                    SR_stats_CR[tst_idx_CR],
                    nbins,
                    reweights=reweights[in_CR][tst_idx_CR],
                    bias=bias_3b_rw_CR,
                )

                tst_result = {
                    "signal_ratio": signal_ratio,
                    "seed": seed,
                    "cheating": cheating,
                    "nbins": nbins,
                    "sigma_avg_SR": hist_info_SR["sigma_avg"],
                    "sigma_avg_bg4b_SR": hist_info_SR["sigma_avg_bg4b"],
                    "sigma_avg_CR": hist_info_CR["sigma_avg"],
                    "sigma_avg_bg4b_CR": hist_info_CR["sigma_avg_bg4b"],
                    "initialize_with_fvt": initialize_with_fvt,
                    "df_SR": hist_info_SR["df"],
                    "df_CR": hist_info_CR["df"],
                    "reweight": rw_method,
                }

                print(
                    {
                        "signal_ratio": signal_ratio,
                        "seed": seed,
                        "nbins": nbins,
                        "cheating": cheating,
                        "sigma_avg_SR": hist_info_SR["sigma_avg"],
                        "sigma_avg_CR": hist_info_CR["sigma_avg"],
                        "reweight": rw_method,
                    }
                )

                tst_results_summary.append(tst_result)
                tst_results_summary_df = pd.concat(
                    [tst_results_summary_df, pd.DataFrame([tst_result])]
                )

            tst_results_summary_df.to_csv(df_name, index=False, sep="\t")
