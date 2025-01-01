import torch
from plots import get_weights_by_sr_stats
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from training_info import TrainingInfo
from dataset import MotherSamples
from events_data import EventsData
from fvt_classifier import FvTClassifier
from attention_classifier import AttentionClassifier

import pandas as pd
from attention_classifier import AttentionClassifier

import logging
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


df_3b = pd.read_hdf("../events/MG3/dataframes/threeTag_picoAOD.h5")
df_bg4b = pd.read_hdf("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
df_signal = pd.read_hdf("../events/MG3/dataframes/HH4b_picoAOD.h5")
df_3b["signal"] = False
df_bg4b["signal"] = False
df_signal["signal"] = True
raw_df_list = [df_3b, df_bg4b, df_signal]

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


def get_events_tst(smeared_hash: str):
    smeared_fvt_tinfo = TrainingInfo.load(smeared_hash)

    base_encoder_hash = smeared_fvt_tinfo.hparams["encoder_hash"]
    base_fvt_tinfo = TrainingInfo.load(base_encoder_hash)
    base_fvt_model = base_fvt_tinfo.load_trained_model("best")
    base_fvt_model.eval()
    base_fvt_model: FvTClassifier

    ms_hash = base_fvt_tinfo.ms_hash
    ms_idx = base_fvt_tinfo.ms_idx
    msamples = MotherSamples.load(ms_hash)
    tst_scdinfo = msamples.scdinfo[~ms_idx]
    df_tst = tst_scdinfo.fetch_data_with_loaded_df(raw_df_list)
    events_tst = EventsData.from_dataframe(df_tst, features)
    return events_tst


def plot_mean_and_fill_between_std(x, y_list, ax, **kwargs):
    y_mean = np.mean(y_list, axis=0)
    y_std = np.std(y_list, axis=0)
    ax.plot(x, y_mean, **kwargs)
    if "color" in kwargs:
        ax.fill_between(
            x, y_mean - y_std, y_mean + y_std, alpha=0.2, color=kwargs["color"]
        )
    else:
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)


n_3b = 100_0000
seeds = range(50)
signal_ratios = [0.005, 0.0075, 0.01, 0.02]
ensemble_seeds = range(15)

for signal_ratio in signal_ratios:
    logger.info(f"signal_ratio={signal_ratio}")
    w_4b_ratio_points = np.arange(0, 1.01, 0.01)
    w_signal_ratio_list = {
        "max_5": [],
        "max_10": [],
        "max_15": [],
        "mean": [],
        "fvt_mean": [],
        "fvt_max_5": [],
        "fvt_max_10": [],
        "fvt_max_15": [],
    }
    for train_seed in ensemble_seeds:
        w_signal_ratio_list[train_seed] = []

    for seed in seeds:
        logger.info(f"seed={seed}")
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
        SR_stats_list = []
        base_fvt_scores_list = []
        for hash in hashes:
            smeared_fvt_tinfo = TrainingInfo.load(hash)
            train_seed = smeared_fvt_tinfo.hparams["train_seed"]
            SR_stats_tst = smeared_fvt_tinfo.aux_info["SR_stats_tst"]
            SR_stats_list.append(SR_stats_tst)
            w_4b_ratio, w_signal_ratio = get_weights_by_sr_stats(
                events_tst, SR_stats_tst
            )
            w_signal_ratio = np.interp(w_4b_ratio_points, w_4b_ratio, w_signal_ratio)
            w_signal_ratio_list[train_seed].append(w_signal_ratio)

            base_fvt_tinfo = TrainingInfo.load(
                smeared_fvt_tinfo.hparams["encoder_hash"]
            )
            base_fvt_model = base_fvt_tinfo.load_trained_model("best")
            base_fvt_model.eval()
            base_fvt_model: FvTClassifier
            base_fvt_scores, _ = base_fvt_model.predict_and_representations(
                events_tst.X_torch
            )
            base_fvt_scores = base_fvt_scores.numpy()[:, 1]
            base_fvt_scores_list.append(base_fvt_scores)

        base_fvt_scores_mean = np.mean(base_fvt_scores_list, axis=0)
        w_4b_ratio, w_signal_ratio = get_weights_by_sr_stats(
            events_tst, base_fvt_scores_mean
        )
        w_signal_ratio = np.interp(w_4b_ratio_points, w_4b_ratio, w_signal_ratio)
        w_signal_ratio_list["fvt_mean"].append(w_signal_ratio)

        base_fvt_scores_max_5 = np.max(base_fvt_scores_list[:5], axis=0)
        w_4b_ratio, w_signal_ratio = get_weights_by_sr_stats(
            events_tst, base_fvt_scores_max_5
        )
        w_signal_ratio = np.interp(w_4b_ratio_points, w_4b_ratio, w_signal_ratio)
        w_signal_ratio_list["fvt_max_5"].append(w_signal_ratio)

        base_fvt_scores_max_10 = np.max(base_fvt_scores_list[:10], axis=0)
        w_4b_ratio, w_signal_ratio = get_weights_by_sr_stats(
            events_tst, base_fvt_scores_max_10
        )
        w_signal_ratio = np.interp(w_4b_ratio_points, w_4b_ratio, w_signal_ratio)
        w_signal_ratio_list["fvt_max_10"].append(w_signal_ratio)

        base_fvt_scores_max_15 = np.max(base_fvt_scores_list[:15], axis=0)
        w_4b_ratio, w_signal_ratio = get_weights_by_sr_stats(
            events_tst, base_fvt_scores_max_15
        )
        w_signal_ratio = np.interp(w_4b_ratio_points, w_4b_ratio, w_signal_ratio)
        w_signal_ratio_list["fvt_max_15"].append(w_signal_ratio)

        SR_stats_max_5 = np.max(SR_stats_list[:5], axis=0)
        w_4b_ratio, w_signal_ratio = get_weights_by_sr_stats(events_tst, SR_stats_max_5)
        w_signal_ratio = np.interp(w_4b_ratio_points, w_4b_ratio, w_signal_ratio)
        w_signal_ratio_list["max_5"].append(w_signal_ratio)

        SR_stats_max_10 = np.max(SR_stats_list[:10], axis=0)
        w_4b_ratio, w_signal_ratio = get_weights_by_sr_stats(
            events_tst, SR_stats_max_10
        )
        w_signal_ratio = np.interp(w_4b_ratio_points, w_4b_ratio, w_signal_ratio)
        w_signal_ratio_list["max_10"].append(w_signal_ratio)

        SR_stats_max_15 = np.max(SR_stats_list[:15], axis=0)
        w_4b_ratio, w_signal_ratio = get_weights_by_sr_stats(
            events_tst, SR_stats_max_15
        )
        w_signal_ratio = np.interp(w_4b_ratio_points, w_4b_ratio, w_signal_ratio)
        w_signal_ratio_list["max_15"].append(w_signal_ratio)

        SR_stats_mean = np.mean(SR_stats_list, axis=0)
        w_4b_ratio, w_signal_ratio = get_weights_by_sr_stats(events_tst, SR_stats_mean)
        w_signal_ratio = np.interp(w_4b_ratio_points, w_4b_ratio, w_signal_ratio)
        w_signal_ratio_list["mean"].append(w_signal_ratio)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_mean_and_fill_between_std(
        w_4b_ratio_points, w_signal_ratio_list[0], ax, label="train_seed=0"
    )
    for key in [
        "mean",
        "max_5",
        "max_10",
        "max_15",
        "fvt_mean",
        "fvt_max_5",
        "fvt_max_10",
        "fvt_max_15",
    ]:
        plot_mean_and_fill_between_std(
            w_4b_ratio_points,
            w_signal_ratio_list[key],
            ax,
            label=key,
            linestyle="--",
        )

    ax.set_title(f"signal_ratio={signal_ratio}")
    ax.legend()
    ax.set_xlabel(r"$P_{4b}(\mathcal{X}_s)$")
    ax.set_ylabel(r"$S(\mathcal{X}_s)$")
    ax.set_aspect("equal")

    plt.savefig(
        f"data/plots/smeared_gamma_ensemble_avg_signal_ratio={signal_ratio}.pdf"
    )
    plt.show()
    plt.close()
