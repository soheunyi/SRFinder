import numpy as np
from constants import FEATURES
from dataset import MotherSamples
from ks_test import max_cdf_diff
from signal_region import compute_sr_stats, get_SR_CR_cut
from training_info import TrainingInfo
from events_data import events_from_scdinfo
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from constants import FEATURES

path_3b = Path("../events/MG3/dataframes/threeTag_picoAOD.h5")
path_4b = Path("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
path_hh4b = Path("../events/MG3/dataframes/HH4b_picoAOD.h5")
path_hh4b_400 = Path("../events/MG3/dataframes/HH4b_400.h5")
path_hh4b_800 = Path("../events/MG3/dataframes/HH4b_800.h5")

df_3b = pd.read_hdf(path_3b)
df_bg4b = pd.read_hdf(path_4b)
df_hh4b = pd.read_hdf(path_hh4b)
df_hh4b_400 = pd.read_hdf(path_hh4b_400)
df_hh4b_800 = pd.read_hdf(path_hh4b_800)

df_3b["signal"] = False
df_bg4b["signal"] = False
df_hh4b["signal"] = True
df_hh4b_400["signal"] = True
df_hh4b_800["signal"] = True

loaded_df = {
    path_3b: df_3b,
    path_4b: df_bg4b,
    path_hh4b: df_hh4b,
    path_hh4b_400: df_hh4b_400,
    path_hh4b_800: df_hh4b_800,
}

experiment_name = "CR_fvt_training_ensemble_max_HH4b_800"
hashes = TrainingInfo.find({"experiment_name": experiment_name})

for hash_ in tqdm(hashes):
    CR_fvt_tinfo = TrainingInfo.load(hash_)
    SR_stats_hashes = CR_fvt_tinfo.hparams["signal_region"]["SR_stats_hashes"]
    ensemble_mode = CR_fvt_tinfo.hparams["signal_region"]["ensemble_mode"]
    stats_type = CR_fvt_tinfo.hparams["signal_region"]["stats_type"]
    seed = CR_fvt_tinfo.hparams["dataset"]["seed"]
    signal_ratio = CR_fvt_tinfo.hparams["dataset"]["signal_ratio"]
    signal_filename = CR_fvt_tinfo.hparams["dataset"]["signal_filename"]

    SR_stats_train, SR_stats_tst = compute_sr_stats(
        SR_stats_hashes,
        signal_filename,
        ensemble_mode,
        stats_type,
    )

    smeared_tinfo = TrainingInfo.load(SR_stats_hashes[0])
    if CR_fvt_tinfo.hparams["signal_region"]["stats_type"] == "smeared":
        noise_scale = smeared_tinfo.hparams["smearing"]["noise_scale"]
    else:
        noise_scale = np.inf
    msamples = MotherSamples.load(smeared_tinfo.ms_hash)
    events_train = events_from_scdinfo(
        msamples.scdinfo[smeared_tinfo.ms_idx], FEATURES, signal_filename
    )
    events_tst = events_from_scdinfo(
        msamples.scdinfo[~smeared_tinfo.ms_idx], FEATURES, signal_filename
    )
    SR_cut, CR_cut = get_SR_CR_cut(
        SR_stats_train, events_train, CR_fvt_tinfo.hparams["signal_region"]
    )
    SR_idx = SR_stats_tst >= SR_cut

    # SR_stats_tst_SR = SR_stats_tst[SR_idx]
    # events_tst_SR = events_tst[SR_idx]
    # fvt_scores_tst_SR = CR_fvt_tinfo.aux_info["fvt_scores_tst_SR"]
    # reweights_tst_SR = fvt_scores_tst_SR / (1 - fvt_scores_tst_SR)
    # rw_tst_SR = np.where(
    #     events_tst_SR.is_4b,
    #     events_tst_SR.weights,
    #     reweights_tst_SR * events_tst_SR.weights,
    # )
    # is_4b_tst_SR = events_tst_SR.is_4b

    # max_diff = max_cdf_diff(
    #     SR_stats_tst_SR[is_4b_tst_SR],
    #     SR_stats_tst_SR[~is_4b_tst_SR],
    #     rw_tst_SR[is_4b_tst_SR],
    #     rw_tst_SR[~is_4b_tst_SR],
    # )

    events_tst_SR = events_tst[SR_idx]
    N_3b_SR = np.sum(events_tst_SR.is_3b)
    N_4b_SR = np.sum(events_tst_SR.is_4b)

    # CR_fvt_tinfo.aux_info.update({"max_cdf_diff": max_diff})
    CR_fvt_tinfo.aux_info.update({"N_3b_SR": N_3b_SR, "N_4b_SR": N_4b_SR})
    CR_fvt_tinfo.save()
