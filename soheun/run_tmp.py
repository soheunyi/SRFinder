import os
import pickle

import numpy as np
import torch
import tqdm
from signal_region import get_SR_CR_cut, get_DRs

from training_info import TrainingInfo
from fvt_classifier import FvTClassifier
from attention_classifier import AttentionClassifier
from events_data import EventsData, events_from_scdinfo
from dataset import MotherSamples
from utils import get_quantiles_with_weights

n_3b = 100_0000
device = torch.device("cuda")
experiment_name = "CR_fvt_training_v2"
signal_filename = "HH4b_picoAOD.h5"
ratio_4b = 0.5
seeds = np.arange(50)
nbins_range = [2**i for i in range(9)]

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


for signal_ratio in [0.0, 0.01, 0.02]:
    hparam_filter = {
        "experiment_name": experiment_name,
        "dataset": lambda x: all(
            [x["seed"] in seeds, x["n_3b"] == n_3b, x["signal_ratio"] == signal_ratio]
        ),
        "aux_info_step": 3,
    }
    hashes = TrainingInfo.find(hparam_filter)

    for hash in tqdm.tqdm(hashes):
        if os.path.exists("./data/tmp/hist_list_v3.pkl"):
            hist_list = pickle.load(open("./data/tmp/hist_list_v3.pkl", "rb"))
            if hash in [v["hash"] for v in hist_list]:
                continue
        else:
            hist_list = []

        CR_fvt_tinfo = TrainingInfo.load(hash)
        smeared_fvt_hash = CR_fvt_tinfo.hparams["smeared_fvt_hash"]
        base_encoder_hash = CR_fvt_tinfo.hparams["encoder_hash"]

        base_fvt_model = TrainingInfo.load(base_encoder_hash).load_trained_model(
            CR_fvt_tinfo.hparams["encoder_mode"]
        )
        base_fvt_model.eval()
        base_fvt_model.to(torch.device("cuda"))

        smeared_fvt_tinfo = TrainingInfo.load(smeared_fvt_hash)
        smeared_fvt_model = smeared_fvt_tinfo.load_trained_model(
            CR_fvt_tinfo.hparams["encoder_mode"]
        )
        smeared_fvt_model.eval()
        smeared_fvt_model.to(torch.device("cuda"))

        # Use the same mother samples and exclude ones used for training base & smeared FvT model
        msamples = MotherSamples.load(smeared_fvt_tinfo.ms_hash)
        events_base_train = events_from_scdinfo(
            msamples.scdinfo[smeared_fvt_tinfo.ms_idx], features, signal_filename
        )
        tst_scdinfo = msamples.scdinfo[~smeared_fvt_tinfo.ms_idx]
        events_tst = events_from_scdinfo(tst_scdinfo, features, signal_filename)
        events_all = EventsData.merge([events_base_train, events_tst])
        gamma_base_all, gamma_smeared_all, q_repr_base_all = get_DRs(
            events_all, base_fvt_model, smeared_fvt_model, return_repr=True
        )
        gamma_base_train = gamma_base_all[: len(events_base_train)]
        gamma_smeared_train = gamma_smeared_all[: len(events_base_train)]
        gamma_base = gamma_base_all[len(events_base_train) :]
        gamma_smeared = gamma_smeared_all[len(events_base_train) :]
        q_repr_train = q_repr_base_all[: len(events_base_train)]
        q_repr_tst = q_repr_base_all[len(events_base_train) :]
        SR_stats_train = np.log(gamma_base_train / gamma_smeared_train)
        SR_stats = np.log(gamma_base / gamma_smeared)

        SR_cut, CR_cut = get_SR_CR_cut(
            SR_stats_train, events_base_train, CR_fvt_tinfo.hparams["signal_region"]
        )

        SR_idx = SR_stats >= SR_cut
        CR_idx = (SR_stats >= CR_cut) & (SR_stats < SR_cut)

        events_tst_SR = events_tst[SR_idx]
        events_tst_CR = events_tst[CR_idx]

        model = CR_fvt_tinfo.hparams["model"]

        CR_fvt_model = CR_fvt_tinfo.load_trained_model("best")
        CR_fvt_model.eval()
        CR_fvt_model.to(torch.device("cuda"))

        if model == "AttentionClassifier":
            CR_fvt_model: AttentionClassifier
            fvt_scores_CR_model = (
                CR_fvt_model.predict(q_repr_tst)[:, 1].detach().cpu().numpy()
            )
        else:
            CR_fvt_model: FvTClassifier
            fvt_scores_CR_model = (
                CR_fvt_model.predict(events_tst.X_torch)[:, 1].detach().cpu().numpy()
            )

        fvt_scores_SR = fvt_scores_CR_model[SR_idx]
        reweights_SR = fvt_scores_SR / (1 - fvt_scores_SR)
        reweights_SR = np.where(events_tst_SR.is_4b, 1, -reweights_SR)

        fvt_scores_CR = fvt_scores_CR_model[CR_idx]
        reweights_CR = fvt_scores_CR / (1 - fvt_scores_CR)
        reweights_CR = np.where(events_tst_CR.is_4b, 1, -reweights_CR)

        SR_stats_SR = SR_stats[SR_idx]
        SR_stats_CR = SR_stats[CR_idx]

        for mode in ["linspace", "quantile"]:
            for nbins in nbins_range:
                if mode == "linspace":
                    bins_SR = np.linspace(SR_cut, np.max(SR_stats_train), nbins + 1)
                    bins_CR = np.linspace(CR_cut, SR_cut, nbins + 1)
                elif mode == "quantile":
                    bins_SR = get_quantiles_with_weights(
                        SR_stats_train[SR_stats_train >= SR_cut],
                        events_base_train.weights[SR_stats_train >= SR_cut],
                        np.linspace(0, 1, nbins + 1),
                    )
                    bins_CR = get_quantiles_with_weights(
                        SR_stats_train[
                            (SR_stats_train < SR_cut) & (SR_stats_train >= CR_cut)
                        ],
                        events_base_train.weights[
                            (SR_stats_train < SR_cut) & (SR_stats_train >= CR_cut)
                        ],
                        np.linspace(0, 1, nbins + 1),
                    )

                hist_diff_SR = np.histogram(
                    SR_stats_SR,
                    bins=bins_SR,
                    weights=reweights_SR * events_tst_SR.weights,
                )[0]
                hist_3b_SR = np.histogram(
                    SR_stats_SR[events_tst_SR.is_3b],
                    bins=bins_SR,
                    weights=events_tst_SR.weights[events_tst_SR.is_3b],
                )[0]
                hist_bg4b_SR = np.histogram(
                    SR_stats_SR[events_tst_SR.is_bg4b],
                    bins=bins_SR,
                    weights=events_tst_SR.weights[events_tst_SR.is_bg4b],
                )[0]
                hist_signal_SR = np.histogram(
                    SR_stats_SR[events_tst_SR.is_signal],
                    bins=bins_SR,
                    weights=events_tst_SR.weights[events_tst_SR.is_signal],
                )[0]
                hist_sq_SR = np.histogram(
                    SR_stats_SR,
                    bins=bins_SR,
                    weights=reweights_SR**2 * events_tst_SR.weights**2,
                )[0]

                hist_diff_CR = np.histogram(
                    SR_stats_CR,
                    bins=bins_CR,
                    weights=reweights_CR * events_tst_CR.weights,
                )[0]
                hist_3b_CR = np.histogram(
                    SR_stats_CR[events_tst_CR.is_3b],
                    bins=bins_CR,
                    weights=events_tst_CR.weights[events_tst_CR.is_3b],
                )[0]
                hist_bg4b_CR = np.histogram(
                    SR_stats_CR[events_tst_CR.is_bg4b],
                    bins=bins_CR,
                    weights=events_tst_CR.weights[events_tst_CR.is_bg4b],
                )[0]
                hist_signal_CR = np.histogram(
                    SR_stats_CR[events_tst_CR.is_signal],
                    bins=bins_CR,
                    weights=events_tst_CR.weights[events_tst_CR.is_signal],
                )[0]
                hist_sq_CR = np.histogram(
                    SR_stats_CR,
                    bins=bins_CR,
                    weights=reweights_CR**2 * events_tst_CR.weights**2,
                )[0]

                hist_list.extend(
                    [
                        {
                            "hash": hash,
                            "model": model,
                            "signal_ratio": signal_ratio,
                            "mode": mode,
                            "nbins": nbins,
                            "diff_SR": hist_diff_SR,
                            "sq_SR": hist_sq_SR,
                            "diff_CR": hist_diff_CR,
                            "sq_CR": hist_sq_CR,
                            "3b_SR": hist_3b_SR,
                            "3b_CR": hist_3b_CR,
                            "bg4b_SR": hist_bg4b_SR,
                            "bg4b_CR": hist_bg4b_CR,
                            "signal_SR": hist_signal_SR,
                            "signal_CR": hist_signal_CR,
                            "bins_SR": bins_SR,
                            "bins_CR": bins_CR,
                        }
                    ]
                )

        with open("./data/tmp/hist_list_v3.pkl", "wb") as f:
            pickle.dump(hist_list, f)
