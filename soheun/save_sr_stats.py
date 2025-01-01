# Saving SR_stats

import numpy as np
import torch
from tqdm import tqdm
from training_info import TrainingInfo
from fvt_classifier import FvTClassifier
from dataset import MotherSamples
from events_data import EventsData

import pandas as pd
from attention_classifier import AttentionClassifier

import logging

# Set up logging
# log time and date for each log message
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

logging.info("Packages loaded")

df_3b = pd.read_hdf("../events/MG3/dataframes/threeTag_picoAOD.h5")
df_bg4b = pd.read_hdf("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
df_signal = pd.read_hdf("../events/MG3/dataframes/HH4b_picoAOD.h5")
df_3b["signal"] = False
df_bg4b["signal"] = False
df_signal["signal"] = True
raw_df_list = [df_3b, df_bg4b, df_signal]

logging.info("Datasets loaded")

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


n_3b = 100_0000
seeds = range(50)
signal_ratios = [0.0075, 0.02]


def get_events_tst(hash: str):
    fvt_tinfo = TrainingInfo.load(hash)
    ms_hash = fvt_tinfo.ms_hash
    ms_idx = fvt_tinfo.ms_idx
    msamples = MotherSamples.load(ms_hash)
    tst_scdinfo = msamples.scdinfo[~ms_idx]
    df_tst = tst_scdinfo.fetch_data_with_loaded_df(raw_df_list)
    events_tst = EventsData.from_dataframe(df_tst, features)
    return events_tst


def calculate_SR_stats(events_tst: EventsData, hash: str):
    smeared_fvt_tinfo = TrainingInfo.load(hash)
    base_encoder_hash = smeared_fvt_tinfo.hparams["encoder_hash"]
    base_fvt_tinfo = TrainingInfo.load(base_encoder_hash)
    base_fvt_model = base_fvt_tinfo.load_trained_model("best")
    base_fvt_model.eval()
    base_fvt_model: FvTClassifier
    base_fvt_scores, base_q_repr = base_fvt_model.predict_and_representations(
        events_tst.X_torch
    )
    base_fvt_scores = base_fvt_scores.numpy()[:, 1]
    base_q_repr = base_q_repr.numpy()
    base_gamma = base_fvt_scores / (1 - base_fvt_scores)
    smeared_fvt_model = smeared_fvt_tinfo.load_trained_model("best")
    smeared_fvt_model.eval()
    smeared_fvt_model.to(torch.device("cuda"))
    smeared_fvt_model: AttentionClassifier
    fvt_smeared = smeared_fvt_model.predict(base_q_repr).numpy()[:, 1]
    gamma_smeared = fvt_smeared / (1 - fvt_smeared)
    SR_stats_tst = np.log(base_gamma / gamma_smeared)
    return SR_stats_tst


for signal_ratio in signal_ratios:
    logging.info(f"Calculating SR stats for signal ratio {signal_ratio}")
    for seed in tqdm(seeds):
        hashes, hparams = TrainingInfo.find(
            {
                "experiment_name": "smeared_fvt_training_ensemble",
                "dataset": lambda x: (
                    x["signal_ratio"] == signal_ratio
                    and x["seed"] == seed
                    and x["n_3b"] == n_3b
                ),
            },
            return_hparams=True,
        )
        # sort hashes by train_seed
        hashes = sorted(hashes, key=lambda x: hparams[hashes.index(x)]["train_seed"])

        events_tst = get_events_tst(hashes[0])
        for hash in hashes:
            smeared_fvt_tinfo = TrainingInfo.load(hash)
            train_seed = smeared_fvt_tinfo.hparams["train_seed"]

            if "SR_stats_tst" in smeared_fvt_tinfo.aux_info.keys():
                continue
            else:
                SR_stats_tst = calculate_SR_stats(events_tst, hash)
                smeared_fvt_tinfo.update_aux_info(SR_stats_tst=SR_stats_tst)
                smeared_fvt_tinfo.save()
                logging.info(f"Saved SR stats for {hash}")
