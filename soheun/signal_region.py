import pathlib
import time
from typing import Literal
import numpy as np
import pandas as pd
import torch
from dataset import MotherSamples
from events_data import EventsData, events_from_scdinfo
from fvt_classifier import FvTClassifier
from attention_classifier import AttentionClassifier
from training_info import TrainingInfo

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


def get_DRs(
    events: EventsData,
    base_fvt_model: FvTClassifier,
    smeared_fvt_model: AttentionClassifier,
    return_repr: bool = False,
) -> tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]:
    """
    Get density ratios from base and smeared models
    """
    X_torch = events.X_torch
    fvt_base_score, q_repr_base = base_fvt_model.predict_and_representations(X_torch)
    fvt_base_score = fvt_base_score[:, 1].cpu().numpy()
    fvt_smeared_score = smeared_fvt_model.predict(q_repr_base)[:, 1].cpu().numpy()
    gamma = fvt_base_score / (1 - fvt_base_score)
    gamma_smeared = fvt_smeared_score / (1 - fvt_smeared_score)
    if return_repr:
        return gamma, gamma_smeared, q_repr_base
    return gamma, gamma_smeared


def get_SR_CR_cut(SR_stats: np.ndarray, events_train: EventsData, SRCR_hparams: dict):
    assert len(SR_stats) == len(events_train)
    assert "4b_in_SR" in SRCR_hparams and "4b_in_CR" in SRCR_hparams

    W_4B_CUT_MIN = 0.001
    W_4B_CUT_MAX = 0.999

    SR_stats_argsort = np.argsort(SR_stats, kind="stable")[::-1]
    SR_stats_sorted = SR_stats[SR_stats_argsort]
    weights = events_train.weights[SR_stats_argsort]
    is_4b = events_train.is_4b[SR_stats_argsort]
    cumul_4b_ratio = np.cumsum(weights * is_4b) / np.sum(weights * is_4b)

    # print("48140", SR_stats_argsort[48140], SR_stats[SR_stats_argsort[48140]])
    # print("48141", SR_stats_argsort[48141], SR_stats[SR_stats_argsort[48141]])
    # print("48142", SR_stats_argsort[48142], SR_stats[SR_stats_argsort[48142]])
    # print("48143", SR_stats_argsort[48143], SR_stats[SR_stats_argsort[48143]])
    # print("48144", SR_stats_argsort[48144], SR_stats[SR_stats_argsort[48144]])
    # print("48145", SR_stats_argsort[48145], SR_stats[SR_stats_argsort[48145]])
    # print("48146", SR_stats_argsort[48146], SR_stats[SR_stats_argsort[48146]])
    # print("48147", SR_stats_argsort[48147], SR_stats[SR_stats_argsort[48147]])
    # print("48148", SR_stats_argsort[48148], SR_stats[SR_stats_argsort[48148]])
    # print(np.sum(is_4b))
    # print("48140", is_4b[48140])
    # print("48141", is_4b[48141])
    # print("48142", is_4b[48142])
    # print("48143", is_4b[48143])
    # print("48144", is_4b[48144])
    # print("48145", is_4b[48145])
    # print("48146", is_4b[48146])
    # print("48147", is_4b[48147])
    # print("48148", is_4b[48148])
    # print(np.sum(weights * is_4b))
    # tmp = np.cumsum(weights * is_4b)
    # print("48140", tmp[48140])
    # print("48141", tmp[48141])
    # print("48142", tmp[48142])
    # print("48143", tmp[48143])
    # print("48144", tmp[48144])
    # print("48145", tmp[48145])
    # print("48146", tmp[48146])
    # print("48147", tmp[48147])
    # print("48148", tmp[48148])

    w_4b_SR_ratio = np.clip(SRCR_hparams["4b_in_SR"], W_4B_CUT_MIN, W_4B_CUT_MAX)
    w_4b_CR_ratio = np.clip(
        SRCR_hparams["4b_in_CR"] + SRCR_hparams["4b_in_SR"], W_4B_CUT_MIN, W_4B_CUT_MAX
    )

    SR_cut, CR_cut = None, None
    for i in range(1, len(cumul_4b_ratio)):
        # use 8 digits precision
        if cumul_4b_ratio[i] > w_4b_SR_ratio and SR_cut is None:
            SR_cut = SR_stats_sorted[i - 1]
        if cumul_4b_ratio[i] > w_4b_CR_ratio and CR_cut is None:
            CR_cut = SR_stats_sorted[i - 1]
        if SR_cut is not None and CR_cut is not None:
            break

    # If the cut is not found, set the cut to the minimum value
    # Both SR and CR cuts should be different
    if SR_cut is None:
        SR_cut = SR_stats_sorted[-1]
    if CR_cut is None:
        CR_cut = SR_stats_sorted[-1]
    if SR_cut == CR_cut:
        raise ValueError("SR and CR cuts are the same")

    return SR_cut, CR_cut


def get_events(
    tinfo: TrainingInfo,
    signal_filename: str,
    loaded_df: dict[pathlib.Path, pd.DataFrame] = {},
):
    ms_hash = tinfo.ms_hash
    ms_idx = tinfo.ms_idx
    msamples = MotherSamples.load(ms_hash)
    train_scdinfo = msamples.scdinfo[ms_idx]
    tst_scdinfo = msamples.scdinfo[~ms_idx]
    events_train = events_from_scdinfo(
        train_scdinfo, features, signal_filename, loaded_df
    )
    events_tst = events_from_scdinfo(tst_scdinfo, features, signal_filename, loaded_df)
    return events_train, events_tst


def get_base_and_smeared_DRs(events_tst: EventsData, smeared_hash: str):
    smeared_tinfo = TrainingInfo.load(smeared_hash)
    smeared_fvt_model = smeared_tinfo.load_trained_model("best")
    smeared_fvt_model.eval()
    smeared_fvt_model.to(torch.device("cuda"))
    smeared_fvt_model: AttentionClassifier

    base_hash = smeared_tinfo.hparams["encoder_hash"]
    base_tinfo = TrainingInfo.load(base_hash)
    base_fvt_model = base_tinfo.load_trained_model("best")
    base_fvt_model.eval()
    base_fvt_model.to(torch.device("cuda"))
    base_fvt_model: FvTClassifier

    base_fvt_score, base_q_repr = base_fvt_model.predict_and_representations(
        events_tst.X_torch
    )
    base_fvt_score = base_fvt_score[:, 1].cpu().numpy()
    gamma_base = base_fvt_score / (1 - base_fvt_score)
    base_q_repr = base_q_repr.cpu().numpy()

    smeared_fvt_score = smeared_fvt_model.predict(base_q_repr)[:, 1].cpu().numpy()
    gamma_smeared = smeared_fvt_score / (1 - smeared_fvt_score)

    return gamma_base, gamma_smeared


def compute_sr_stats(
    hashes: list[str],
    signal_filename: str,
    ensemble_mode: Literal["mean", "max"] = "max",
    stats_type: Literal["fvt", "smeared"] = "smeared",
):
    smeared_tinfo_0 = TrainingInfo.load(hashes[0])
    smeared_tinfo_list: list[TrainingInfo] = []

    for hash in hashes:
        tinfo = TrainingInfo.load(hash)
        assert tinfo.aux_info["step"] == 2, f"{hash}: Step is {tinfo.aux_info['step']}"
        assert (
            tinfo.ms_hash == smeared_tinfo_0.ms_hash
        ), f"{hash}: MS hash is not matching"
        assert np.all(
            tinfo.ms_idx == smeared_tinfo_0.ms_idx
        ), f"{hash}: MS idx is not matching"
        smeared_tinfo_list.append(tinfo)
        tinfo_signal_filename = tinfo.hparams["dataset"]["signal_filename"]
        assert (
            tinfo_signal_filename == signal_filename
        ), f"{hash}: Signal filename is not matching"

    SR_stats_tst_list = []
    SR_stats_train_list = []

    for tinfo in smeared_tinfo_list:
        base_hash = tinfo.hparams["encoder_hash"]
        base_tinfo = TrainingInfo.load(base_hash)
        assert (
            "base_fvt_logit_train" in base_tinfo.aux_info
        ), f"{base_hash}: base_fvt_logit_train is not found. Run step_1_save_aux_info first"
        assert (
            "base_fvt_logit_tst" in base_tinfo.aux_info
        ), f"{base_hash}: base_fvt_logit_tst is not found. Run step_1_save_aux_info first"

        base_fvt_logit_train = base_tinfo.aux_info["base_fvt_logit_train"]
        base_fvt_logit_tst = base_tinfo.aux_info["base_fvt_logit_tst"]

        if stats_type == "smeared":
            assert (
                "smeared_fvt_logit_train" in tinfo.aux_info
            ), f"{tinfo.hash}: smeared_fvt_logit_train is not found. Run step_2_save_aux_info first"
            assert (
                "smeared_fvt_logit_tst" in tinfo.aux_info
            ), f"{tinfo.hash}: smeared_fvt_logit_tst is not found. Run step_2_save_aux_info first"
            smeared_fvt_logit_train = tinfo.aux_info["smeared_fvt_logit_train"]
            smeared_fvt_logit_tst = tinfo.aux_info["smeared_fvt_logit_tst"]
            SR_stats_train = base_fvt_logit_train - smeared_fvt_logit_train
            SR_stats_tst = base_fvt_logit_tst - smeared_fvt_logit_tst
        elif stats_type == "fvt":
            SR_stats_train = base_fvt_logit_train
            SR_stats_tst = base_fvt_logit_tst
        else:
            raise ValueError(f"stats_type {stats_type} not supported")

        SR_stats_train_list.append(SR_stats_train)
        SR_stats_tst_list.append(SR_stats_tst)

    if ensemble_mode == "mean":
        ensemble_SR_stats_train = np.mean(SR_stats_train_list, axis=0)
        ensemble_SR_stats_tst = np.mean(SR_stats_tst_list, axis=0)
    elif ensemble_mode == "max":
        ensemble_SR_stats_train = np.max(SR_stats_train_list, axis=0)
        ensemble_SR_stats_tst = np.max(SR_stats_tst_list, axis=0)
    else:
        raise ValueError(f"ensemble_mode {ensemble_mode} not supported")

    return ensemble_SR_stats_train, ensemble_SR_stats_tst
