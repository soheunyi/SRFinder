from typing import Literal

import numpy as np
import torch

from attention_classifier import AttentionClassifier
from events_data import EventsData
from dataset import MotherSamples
from fvt_classifier import FvTClassifier

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


def get_events_tst(hash: str):
    fvt_tinfo = TrainingInfo.load(hash)
    ms_hash = fvt_tinfo.ms_hash
    ms_idx = fvt_tinfo.ms_idx
    msamples = MotherSamples.load(ms_hash)
    tst_scdinfo = msamples.scdinfo[~ms_idx]
    df_tst = tst_scdinfo.fetch_data()
    events_tst = EventsData.from_dataframe(df_tst, features)
    return events_tst


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
    ensemble_mode: Literal["mean", "max"] | None = "max",
    stats_type: Literal["fvt", "smeared"] = "smeared",
):
    assert ensemble_mode is not None or len(hashes) == 1
    tinfo_0 = TrainingInfo.load(hashes[0])
    events_tst = get_events_tst(hashes[0])

    tinfo_list: list[TrainingInfo] = []
    for hash in hashes:
        tinfo = TrainingInfo.load(hash)
        assert tinfo.aux_info["step"] == 2
        assert tinfo.ms_hash == tinfo_0.ms_hash
        assert np.all(tinfo.ms_idx == tinfo_0.ms_idx)
        tinfo_list.append(tinfo)

    sr_stats_tst_list = []
    if stats_type == "smeared":
        for tinfo in tinfo_list:
            if "SR_stats_tst" in tinfo.aux_info:
                SR_stats_tst = tinfo.aux_info["SR_stats_tst"]
            else:
                gamma_base, gamma_smeared = get_base_and_smeared_DRs(
                    events_tst, tinfo.hash
                )
                SR_stats_tst = np.log(gamma_base / gamma_smeared)
                tinfo.update_aux_info(SR_stats_tst=SR_stats_tst)
                tinfo.save()
            sr_stats_tst_list.append(SR_stats_tst)

    elif stats_type == "fvt":
        for tinfo in tinfo_list:
            base_hash = tinfo.hparams["encoder_hash"]
            base_tinfo = TrainingInfo.load(base_hash)
            base_fvt_model = base_tinfo.load_trained_model("best")
            base_fvt_model.eval()
            base_fvt_model.to(torch.device("cuda"))
            base_fvt_model: FvTClassifier
            base_fvt_score = (
                base_fvt_model.predict(events_tst.X_torch)[:, 1].cpu().numpy()
            )
            sr_stats_tst_list.append(base_fvt_score)

    else:
        raise ValueError(f"stats_type {stats_type} not supported")

    if ensemble_mode == "mean":
        sr_stats_tst = np.mean(sr_stats_tst_list, axis=0)
    elif ensemble_mode == "max":
        sr_stats_tst = np.max(sr_stats_tst_list, axis=0)
    else:
        raise ValueError(f"ensemble_mode {ensemble_mode} not supported")

    return sr_stats_tst
