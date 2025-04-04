import numpy as np
from dataset import MotherSamples
from training_info import TrainingInfo
from events_data import EventsData, get_is_signal
from signal_region import compute_sr_stats, get_SR_CR_cut
import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path

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

path_3b = Path("../events/MG3/dataframes/threeTag_picoAOD.h5")
path_4b = Path("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
path_signal = Path("../events/MG3/dataframes/HH4b_picoAOD.h5")
df_3b = pd.read_hdf(path_3b)
df_bg4b = pd.read_hdf(path_4b)
df_signal = pd.read_hdf(path_signal)
df_3b["signal"] = False
df_bg4b["signal"] = False
df_signal["signal"] = True
raw_df_list = [df_3b, df_bg4b, df_signal]
loaded_df = {path_3b: df_3b, path_4b: df_bg4b, path_signal: df_signal}


experiment_name = "CR_fvt_training_ensemble_max_smeared"
n_3b = 100_0000
ratio_4b = 0.5
signal_filename = "HH4b_picoAOD.h5"
nbins_list = [4, 8, 16, 32, 64]
pull_dict = []
hists_dicts_loaded = [
    pickle.load(
        open(
            f"./data/tmp/test_info_by_hashes_order_{correction_order}_mi_test.pkl", "rb"
        )
    )
    for correction_order in [0, 1, 2]
]

for SR_size in [0.05, 0.1, 0.15, 0.2]:
    CR_size = 1 - SR_size
    for signal_ratio in [0.0, 0.005, 0.0075, 0.01, 0.02]:
        hparams_filter = {
            "experiment_name": experiment_name,
            "aux_info_step": 3,
            "dataset": lambda x: (
                x["n_3b"] == n_3b
                and x["ratio_4b"] == ratio_4b
                and x["signal_filename"] == signal_filename
                and x["signal_ratio"] == signal_ratio
            ),
            "signal_region": lambda x: (
                x["4b_in_SR"] == SR_size and x["4b_in_CR"] == CR_size
            ),
        }
        CR_fvt_hashes = TrainingInfo.find(hparams_filter)
        for CR_fvt_hash in tqdm(CR_fvt_hashes):
            CR_fvt_tinfo = TrainingInfo.load(CR_fvt_hash)
            SR_stats_hashes = CR_fvt_tinfo.hparams["signal_region"]["SR_stats_hashes"]
            ensemble_mode = CR_fvt_tinfo.hparams["signal_region"]["ensemble_mode"]
            stats_type = CR_fvt_tinfo.hparams["signal_region"]["stats_type"]
            signal_filename = CR_fvt_tinfo.hparams["dataset"]["signal_filename"]

            ms_idx = TrainingInfo.load(SR_stats_hashes[0]).ms_idx
            msamples = MotherSamples.load(CR_fvt_tinfo.ms_hash)
            train_scdinfo = msamples.scdinfo[ms_idx]
            tst_scdinfo = msamples.scdinfo[~ms_idx]

            df_train = train_scdinfo.fetch_data(loaded_df)
            df_train["signal"] = get_is_signal(train_scdinfo, signal_filename)
            events_train = EventsData.from_dataframe(df_train, features)

            df_tst = tst_scdinfo.fetch_data(loaded_df)
            df_tst["signal"] = get_is_signal(tst_scdinfo, signal_filename)
            events_tst = EventsData.from_dataframe(df_tst, features)

            SR_stats_train, SR_stats_tst = compute_sr_stats(
                SR_stats_hashes,
                signal_filename,
                ensemble_mode,
                stats_type,
            )

            SR_cut, CR_cut = get_SR_CR_cut(
                SR_stats_train, events_train, CR_fvt_tinfo.hparams["signal_region"]
            )
            # SR_idx = SR_stats_tst >= SR_cut
            CR_idx = (SR_stats_tst < SR_cut) & (SR_stats_tst >= CR_cut)
            events_tst_CR = events_tst[CR_idx]
            weights_4b_tst_CR = events_tst_CR.total_weight_4b
            weights_3b_tst_CR = (
                events_tst_CR.total_weight - events_tst_CR.total_weight_4b
            )

            mi_test_hashes = TrainingInfo.find({"CR_fvt_hash": CR_fvt_hash})
            for mi_test_hash in mi_test_hashes:
                mi_test_tinfo = TrainingInfo.load(mi_test_hash)
                batch_size = mi_test_tinfo.hparams["dataloader"]["batch_size"]
                for correction_order in [0, 1, 2]:
                    hists_dict = hists_dicts_loaded[correction_order]
                    if mi_test_hash not in hists_dict:
                        continue
                    hists = hists_dict[mi_test_hash]["hists"]
                    corrections = hists_dict[mi_test_hash]["corrections"]

                    for nbins in nbins_list:
                        hist_3b_corrected = hists[nbins]["3b_corrected"]
                        hist_4b = hists[nbins]["4b"]
                        hist_4b_sq = hists[nbins]["4b_sq"]
                        hist_3b_sq = hists[nbins]["3b_corrected_sq"]
                        hist_signal = hists[nbins]["signal"]
                        hist_bg4b = hists[nbins]["bg4b"]
                        n_bins_eff = corrections[nbins][0][2]

                        v_no_shape = hist_4b_sq + hist_3b_sq
                        pull_no_shape = (hist_4b - hist_3b_corrected) / np.sqrt(
                            v_no_shape
                        )
                        v_shape = (
                            v_no_shape
                            + (1 / weights_3b_tst_CR + 1 / weights_4b_tst_CR)
                            * hist_3b_corrected**2
                        )
                        pull_shape = (hist_4b - hist_3b_corrected) / np.sqrt(v_shape)
                        pull_bg4b = (hist_bg4b - hist_3b_corrected) / np.sqrt(
                            v_no_shape
                        )
                        pull_signal = hist_signal / np.sqrt(v_no_shape)
                        mean_pull_sq_no_shape = np.sqrt(np.mean(pull_no_shape**2))
                        mean_pull_sq_shape = np.sqrt(np.mean(pull_shape**2))

                        pull_dict.append(
                            {
                                "mi_test_hash": mi_test_hash,
                                "SR_size": SR_size,
                                "signal_ratio": signal_ratio,
                                "batch_size": batch_size,
                                "nbins": nbins,
                                "mean_pull_sq_no_shape": mean_pull_sq_no_shape,
                                "mean_pull_sq_shape": mean_pull_sq_shape,
                                "n_bins_eff": n_bins_eff,
                                "correction_order": correction_order,
                                "pull_signal": pull_signal,
                                "pull_bg4b": pull_bg4b,
                                "pull_no_shape": pull_no_shape,
                                "pull_shape": pull_shape,
                            }
                        )

with open(f"./data/tmp/pull_dict_mi_test.pkl", "wb") as f:
    pickle.dump(pull_dict, f)
