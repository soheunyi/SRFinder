import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch

from plots import hist_events_by_labels
from events_data import EventsData
from fvt_classifier import FvTClassifier
from tst_info import TSTInfo

# import LogNorm
from matplotlib.colors import LogNorm


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

from itertools import product
from training_info import TrainingInfo
from plots import calibration_plot, plot_rewighted_samples_by_model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ancillary_features import get_m4j
from pl_callbacks import CalibrationPlotCallback, ReweightedPlotCallback

# use tex
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["figure.labelsize"] = 20
plt.rcParams["lines.markersize"] = 3


def get_SR_CR_cut(SR_stats: np.ndarray, events_tst: EventsData, SRCR_hparams: dict):
    assert len(SR_stats) == len(events_tst)
    assert "4b_in_SR" in SRCR_hparams and "4b_in_CR" in SRCR_hparams

    W_4B_CUT_MIN = 0.001
    W_4B_CUT_MAX = 0.999

    SR_stats_argsort = np.argsort(SR_stats)[::-1]
    SR_stats_sorted = SR_stats[SR_stats_argsort]
    weights = events_tst.weights[SR_stats_argsort]
    is_4b = events_tst.is_4b[SR_stats_argsort]
    cumul_4b_ratio = np.cumsum(weights * is_4b) / np.sum(weights * is_4b)

    w_4b_SR_ratio = np.clip(SRCR_hparams["4b_in_SR"], W_4B_CUT_MIN, W_4B_CUT_MAX)
    w_4b_CR_ratio = np.clip(
        SRCR_hparams["4b_in_CR"] + SRCR_hparams["4b_in_SR"], W_4B_CUT_MIN, W_4B_CUT_MAX
    )

    SR_cut, CR_cut = None, None
    for i in range(1, len(cumul_4b_ratio)):
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


fvt_config = {
    "model": "FvTClassifier",
    "dim_dijet_features": 6,
    "dim_quadjet_features": 6,
    "depth": {"encoder": 4, "decoder": 1},
    "fit_batch_size": 1024,
    "model_seed": 0,
    "train_seed": 0,
    "data_seed": 0,
    "max_epochs": 100,
    "val_ratio": 0.33,
    "early_stop_patience": None,
    "optimizer": {"type": "Adam", "lr": 0.01},
    "lr_scheduler": {
        "type": "ReduceLROnPlateau",
        "factor": 0.5,
        "threshold": 0.0001,
        "patience": 10,
        "cooldown": 1,
        "min_lr": 0.0002,
    },
    # "dataloader": {
    #     "batch_size": 1024 * 2**5,
    #     # "batch_size_multiplier": 2,
    #     # "batch_size_milestones": [1, 3, 6, 10, 15],
    # },
    "dataloader": {
        "batch_size": 1024,
        "batch_size_multiplier": 2,
        "batch_size_milestones": [1, 3, 6, 10, 15],
    },
}

att_config = {
    "model": "AttentionClassifier",
    "dim_q": 6,
    "num_classes": 2,
    "depth": 8,
    "fit_batch_size": 1024,
    "model_seed": 0,
    "train_seed": 0,
    "data_seed": 0,
    "max_epochs": 30,
    "val_ratio": 0.33,
    "early_stop_patience": None,
    "optimizer": {"type": "Adam", "lr": 0.01},
    "lr_scheduler": {
        "type": "ReduceLROnPlateau",
        "factor": 0.25,
        "threshold": 0.0001,
        "patience": 3,
        "cooldown": 1,
        "min_lr": 0.0002,
    },
    "dataloader": {
        "batch_size": 1024,
        "batch_size_multiplier": 2,
        "batch_size_milestones": [1, 3, 6, 10, 15],
    },
}

import time
import pytorch_lightning as pl
import yaml
from events_data import events_from_scdinfo
from dataset import MotherSamples
from training_info import TrainingInfo
from attention_classifier import AttentionClassifier
from plots import plot_sr_stats
import tqdm
from torch.utils.data import TensorDataset

n_3b = 100_0000
device = torch.device("cuda")
experiment_name = "smeared_fvt_training"
signal_filename = "HH4b_picoAOD.h5"
ratio_4b = 0.5
seeds = np.arange(10)
# seeds = [3, 6, 9]
signal_ratio = 0.0

hparam_filter = {
    "experiment_name": lambda x: x in [experiment_name],
    "dataset": lambda x: all(
        [x["seed"] in seeds, x["n_3b"] == n_3b, x["signal_ratio"] == signal_ratio]
    ),
    "model": "AttentionClassifier",
}
hashes = TrainingInfo.find(hparam_filter)

SRCR_hparams = {"4b_in_SR": 0.05, "4b_in_CR": 0.95}
cal_bins = np.linspace(0, 1, 21)
quantiles = np.linspace(0, 1, 21)

for tinfo_hash in hashes:
    smeared_tinfo = TrainingInfo.load(tinfo_hash)
    seed = smeared_tinfo.hparams["dataset"]["seed"]
    smeared_fvt = smeared_tinfo.load_trained_model("best")
    smeared_fvt.eval()
    smeared_fvt.to(device)

    base_tinfo = TrainingInfo.load(smeared_tinfo.hparams["encoder_hash"])
    loaded = torch.load(f"./data/checkpoints/{base_tinfo.hash}_best.ckpt")
    base_fvt = base_tinfo.load_trained_model("best")
    base_fvt.eval()
    base_fvt.to(device)

    msamples = MotherSamples.load(smeared_tinfo.ms_hash)
    tst_scdinfo = msamples.scdinfo[~smeared_tinfo.ms_idx]
    events_tst = events_from_scdinfo(tst_scdinfo, features, signal_filename)
    # Should not be shuffled

    probs_4b_base = base_fvt.predict(events_tst.X_torch)[:, 1].detach().cpu().numpy()
    q_repr_tst = base_fvt.representations(events_tst.X_torch)[0]
    probs_4b_smeared = smeared_fvt.predict(q_repr_tst)[:, 1].detach().cpu().numpy()
    gamma_base = probs_4b_base / (1 - probs_4b_base)
    gamma_smeared = probs_4b_smeared / (1 - probs_4b_smeared)

    SR_stats = gamma_base / gamma_smeared

    SR_cut, CR_cut = get_SR_CR_cut(SR_stats, events_tst, SRCR_hparams)

    SR_idx = SR_stats >= SR_cut
    CR_idx = (SR_stats >= CR_cut) & (SR_stats < SR_cut)

    events_tst_SR = events_tst[SR_idx]
    events_tst_CR = events_tst[CR_idx]

    # smeared_tinfo.ms_idx: boolean indexing -> integer indexing
    tst_ms_idx = ~smeared_tinfo.ms_idx
    tst_ms_idx_int = tst_ms_idx.nonzero()[0]
    CR_ms_idx_int = tst_ms_idx_int[CR_idx]
    CR_ms_idx_bool = np.zeros_like(tst_ms_idx, dtype=bool)
    CR_ms_idx_bool[CR_ms_idx_int] = True

    CR_fvt_tinfo = TrainingInfo(
        hparams=fvt_config, ms_hash=smeared_tinfo.ms_hash, ms_idx=CR_ms_idx_bool
    )

    # CR_att_clf_tinfo = TrainingInfo(
    #     hparams=att_config, ms_hash=smeared_tinfo.ms_hash, ms_idx=CR_ms_idx_bool
    # )

    model_seed = seed
    pl.seed_everything(model_seed)

    # CR_fvt_run_name = f"CR_fvt_seed_{seed}_CR_{SRCR_hparams['4b_in_CR']}_SR_{SRCR_hparams['4b_in_SR']}_signal_ratio_{signal_ratio}_from_base_model"
    # CR_fvt = FvTClassifier.load_from_checkpoint(
    #     f"./data/checkpoints/{base_tinfo.hash}_best.ckpt"
    # )

    CR_fvt_run_name = f"CR_fvt_seed_{seed}_CR_{SRCR_hparams['4b_in_CR']}_SR_{SRCR_hparams['4b_in_SR']}_signal_ratio_{signal_ratio}"
    CR_fvt = FvTClassifier(
        num_classes=2,
        dim_input_jet_features=4,
        dim_dijet_features=fvt_config["dim_dijet_features"],
        dim_quadjet_features=fvt_config["dim_quadjet_features"],
        run_name=CR_fvt_run_name,
        device=device,
        depth=fvt_config["depth"],
    )

    # CR_att_clf_run_name = f"CR_att_clf_seed_{seed}_CR_{SRCR_hparams['4b_in_CR']}_SR_{SRCR_hparams['4b_in_SR']}_signal_ratio_{signal_ratio}"
    # CR_att_clf = AttentionClassifier(
    #     dim_q=att_config["dim_q"],
    #     num_classes=att_config["num_classes"],
    #     run_name=CR_att_clf_run_name,
    #     depth=att_config["depth"],
    # )

    train_seed = seed
    CR_fvt_train, CR_fvt_val = CR_fvt_tinfo.fetch_train_val_tensor_datasets(
        features=features, label="fourTag", weight="weight", label_dtype=torch.long
    )

    # q_repr_train = base_fvt.representations(CR_fvt_train.tensors[0])[0]
    # q_repr_val = base_fvt.representations(CR_fvt_val.tensors[0])[0]

    # CR_att_clf_train = TensorDataset(
    #     q_repr_train, CR_fvt_train.tensors[1], CR_fvt_train.tensors[2]
    # )
    # CR_att_clf_val = TensorDataset(
    #     q_repr_val, CR_fvt_val.tensors[1], CR_fvt_val.tensors[2]
    # )

    # CR_att_clf.fit(
    #     CR_att_clf_train,
    #     CR_att_clf_val,
    #     max_epochs=att_config["max_epochs"],
    #     train_seed=train_seed,
    #     save_checkpoint=False,
    #     callbacks=[],
    #     tb_log_dir="CR_ablation",
    #     optimizer_config=att_config["optimizer"],
    #     lr_scheduler_config=att_config["lr_scheduler"],
    #     early_stop_patience=att_config["early_stop_patience"],
    #     dataloader_config=att_config["dataloader"],
    # )

    # CR_att_clf = AttentionClassifier.load_from_checkpoint(
    #     f"./data/tmp/checkpoints/{CR_att_clf_run_name}_best.ckpt"
    # )
    # CR_att_clf.eval()
    # CR_att_clf.to(device)

    # q_repr_tst_SR = base_fvt.representations(events_tst_SR.X_torch)[0]
    # q_repr_tst_CR = base_fvt.representations(events_tst_CR.X_torch)[0]

    # fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    # fig.suptitle(
    #     f"CR_att_clf, seed={seed}, CR={SRCR_hparams['4b_in_CR']}, SR={SRCR_hparams['4b_in_SR']}, signal_ratio={signal_ratio}"
    # )
    # fig.supxlabel(rf"SR Stats ($\gamma / \tilde\gamma$)")
    # ax[0].set_title(f"Control region")
    # ax[1].set_title(f"Signal region")

    # fvt_scores_SR = CR_att_clf.predict(q_repr_tst_SR)[:, 1].detach().cpu().numpy()
    # fvt_scores_CR = CR_att_clf.predict(q_repr_tst_CR)[:, 1].detach().cpu().numpy()

    # CR_cal_bins = np.linspace(np.min(SR_stats[CR_idx]), np.max(SR_stats[CR_idx]), 20)
    # SR_cal_bins = np.linspace(np.min(SR_stats[SR_idx]), np.max(SR_stats[SR_idx]), 20)

    # plot_rewighted_samples_by_model(
    #     events_tst_CR,
    #     x_values=SR_stats[CR_idx],
    #     fvt_scores=fvt_scores_CR,
    #     bins=CR_cal_bins,
    #     figsize=(10, 5),
    #     ratio_4b=ratio_4b,
    #     ax=ax[0],
    # )
    # ax[0].hist(
    #     SR_stats[CR_idx][events_tst_CR.is_3b],
    #     bins=CR_cal_bins,
    #     label="3b",
    #     histtype="step",
    #     linestyle="--",
    #     weights=events_tst_CR.weights[events_tst_CR.is_3b],
    # )
    # ax[0].legend()

    # plot_rewighted_samples_by_model(
    #     events_tst_SR,
    #     x_values=SR_stats[SR_idx],
    #     fvt_scores=fvt_scores_SR,
    #     bins=SR_cal_bins,
    #     figsize=(10, 5),
    #     ratio_4b=ratio_4b,
    #     ax=ax[1],
    # )
    # ax[1].hist(
    #     SR_stats[SR_idx][events_tst_SR.is_3b],
    #     bins=SR_cal_bins,
    #     label="3b",
    #     histtype="step",
    #     linestyle="--",
    #     weights=events_tst_SR.weights[events_tst_SR.is_3b],
    # )
    # ax[1].legend()

    # plt.tight_layout()
    # plt.savefig(
    #     f"./data/plots/CR_att_clf_seed_{seed}_CR_{SRCR_hparams['4b_in_CR']}_SR_{SRCR_hparams['4b_in_SR']}_signal_ratio_{signal_ratio}_SR_stats.pdf",
    #     dpi=300,
    # )
    # plt.close("all")

    # fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    # fig.suptitle(
    #     f"CR_att_clf, seed={seed}, CR={SRCR_hparams['4b_in_CR']}, SR={SRCR_hparams['4b_in_SR']}, signal_ratio={signal_ratio}"
    # )
    # fig.supxlabel(rf"FvT score (prob 4b)")
    # ax[0].set_title(f"Control region")
    # ax[1].set_title(f"Signal region")

    # CR_cal_bins = np.linspace(np.min(fvt_scores_CR), np.max(fvt_scores_CR), 20)
    # SR_cal_bins = np.linspace(np.min(fvt_scores_SR), np.max(fvt_scores_SR), 20)

    # plot_rewighted_samples_by_model(
    #     events_tst_CR,
    #     x_values=fvt_scores_CR,
    #     fvt_scores=fvt_scores_CR,
    #     bins=CR_cal_bins,
    #     figsize=(10, 5),
    #     ratio_4b=ratio_4b,
    #     ax=ax[0],
    # )
    # ax[0].hist(
    #     fvt_scores_CR[events_tst_CR.is_3b],
    #     bins=CR_cal_bins,
    #     label="3b",
    #     histtype="step",
    #     linestyle="--",
    #     weights=events_tst_CR.weights[events_tst_CR.is_3b],
    # )
    # ax[0].legend()

    # plot_rewighted_samples_by_model(
    #     events_tst_SR,
    #     x_values=fvt_scores_SR,
    #     fvt_scores=fvt_scores_SR,
    #     bins=SR_cal_bins,
    #     figsize=(10, 5),
    #     ratio_4b=ratio_4b,
    #     ax=ax[1],
    # )
    # ax[1].hist(
    #     fvt_scores_SR[events_tst_SR.is_3b],
    #     bins=SR_cal_bins,
    #     label="3b",
    #     histtype="step",
    #     linestyle="--",
    #     weights=events_tst_SR.weights[events_tst_SR.is_3b],
    # )
    # ax[1].legend()

    # plt.tight_layout()
    # plt.savefig(
    #     f"./data/plots/CR_att_clf_seed_{seed}_CR_{SRCR_hparams['4b_in_CR']}_SR_{SRCR_hparams['4b_in_SR']}_signal_ratio_{signal_ratio}_FvT_score.pdf",
    #     dpi=300,
    # )
    # plt.close("all")

    CR_fvt.run_name = CR_fvt_run_name

    CR_fvt.fit(
        CR_fvt_train,
        CR_fvt_val,
        max_epochs=fvt_config["max_epochs"],
        train_seed=train_seed,
        save_checkpoint=False,
        callbacks=[],
        tb_log_dir="CR_ablation",
        optimizer_config=fvt_config["optimizer"],
        lr_scheduler_config=fvt_config["lr_scheduler"],
        early_stop_patience=fvt_config["early_stop_patience"],
        dataloader_config=fvt_config["dataloader"],
    )

    CR_fvt = FvTClassifier.load_from_checkpoint(
        f"./data/tmp/checkpoints/{CR_fvt_run_name}_best.ckpt"
    )
    CR_fvt.eval()
    CR_fvt.to(device)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"CR_fvt, seed={seed}, CR={SRCR_hparams['4b_in_CR']}, SR={SRCR_hparams['4b_in_SR']}, signal_ratio={signal_ratio}"
    )
    fig.supxlabel(rf"SR Stats ($\gamma / \tilde\gamma$)")
    ax[0].set_title(f"Control region")
    ax[1].set_title(f"Signal region")

    fvt_scores_SR = CR_fvt.predict(events_tst_SR.X_torch)[:, 1].detach().cpu().numpy()
    fvt_scores_CR = CR_fvt.predict(events_tst_CR.X_torch)[:, 1].detach().cpu().numpy()

    CR_cal_bins = np.linspace(np.min(SR_stats[CR_idx]), np.max(SR_stats[CR_idx]), 20)
    SR_cal_bins = np.linspace(np.min(SR_stats[SR_idx]), np.max(SR_stats[SR_idx]), 20)

    plot_rewighted_samples_by_model(
        events_tst_CR,
        x_values=SR_stats[CR_idx],
        fvt_scores=fvt_scores_CR,
        bins=CR_cal_bins,
        figsize=(10, 5),
        ratio_4b=ratio_4b,
        ax=ax[0],
    )
    ax[0].hist(
        SR_stats[CR_idx][events_tst_CR.is_3b],
        bins=CR_cal_bins,
        label="3b",
        histtype="step",
        linestyle="--",
        weights=events_tst_CR.weights[events_tst_CR.is_3b],
    )
    ax[0].legend()

    plot_rewighted_samples_by_model(
        events_tst_SR,
        x_values=SR_stats[SR_idx],
        fvt_scores=fvt_scores_SR,
        bins=SR_cal_bins,
        figsize=(10, 5),
        ratio_4b=ratio_4b,
        ax=ax[1],
    )
    ax[1].hist(
        SR_stats[SR_idx][events_tst_SR.is_3b],
        bins=SR_cal_bins,
        label="3b",
        histtype="step",
        linestyle="--",
        weights=events_tst_SR.weights[events_tst_SR.is_3b],
    )
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(
        # f"./data/plots/CR_fvt_seed_{seed}_CR_{SRCR_hparams['4b_in_CR']}_SR_{SRCR_hparams['4b_in_SR']}_signal_ratio_{signal_ratio}_from_base_model_SR_stats.pdf",
        f"./data/plots/CR_fvt_seed_{seed}_CR_{SRCR_hparams['4b_in_CR']}_SR_{SRCR_hparams['4b_in_SR']}_signal_ratio_{signal_ratio}_SR_stats.pdf",
        dpi=300,
    )
    plt.close("all")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"CR_fvt, seed={seed}, CR={SRCR_hparams['4b_in_CR']}, SR={SRCR_hparams['4b_in_SR']}, signal_ratio={signal_ratio}"
    )
    fig.supxlabel(rf"FvT score (prob 4b)")
    ax[0].set_title(f"Control region")
    ax[1].set_title(f"Signal region")

    CR_cal_bins = np.linspace(np.min(fvt_scores_CR), np.max(fvt_scores_CR), 20)
    SR_cal_bins = np.linspace(np.min(fvt_scores_SR), np.max(fvt_scores_SR), 20)

    plot_rewighted_samples_by_model(
        events_tst_CR,
        x_values=fvt_scores_CR,
        fvt_scores=fvt_scores_CR,
        bins=CR_cal_bins,
        figsize=(10, 5),
        ratio_4b=ratio_4b,
        ax=ax[0],
    )
    ax[0].hist(
        fvt_scores_CR[events_tst_CR.is_3b],
        bins=CR_cal_bins,
        label="3b",
        histtype="step",
        linestyle="--",
        weights=events_tst_CR.weights[events_tst_CR.is_3b],
    )
    ax[0].legend()

    plot_rewighted_samples_by_model(
        events_tst_SR,
        x_values=fvt_scores_SR,
        fvt_scores=fvt_scores_SR,
        bins=SR_cal_bins,
        figsize=(10, 5),
        ratio_4b=ratio_4b,
        ax=ax[1],
    )
    ax[1].hist(
        fvt_scores_SR[events_tst_SR.is_3b],
        bins=SR_cal_bins,
        label="3b",
        histtype="step",
        linestyle="--",
        weights=events_tst_SR.weights[events_tst_SR.is_3b],
    )
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(
        # f"./data/plots/CR_fvt_seed_{seed}_CR_{SRCR_hparams['4b_in_CR']}_SR_{SRCR_hparams['4b_in_SR']}_signal_ratio_{signal_ratio}_from_base_model_FvT_score.pdf",
        f"./data/plots/CR_fvt_seed_{seed}_CR_{SRCR_hparams['4b_in_CR']}_SR_{SRCR_hparams['4b_in_SR']}_signal_ratio_{signal_ratio}_FvT_score.pdf",
        dpi=300,
    )
    plt.close("all")
