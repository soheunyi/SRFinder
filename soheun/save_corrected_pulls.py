import numpy as np
import pandas as pd
import os

import tqdm
from training_info import TrainingInfo
import multiprocessing as mp
import pickle
from correct_systematic_error import correct_systematic_error
from concurrent.futures import ProcessPoolExecutor


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

print("Loading dataframes")
df_3b = pd.read_hdf("../events/MG3/dataframes/threeTag_picoAOD.h5")
df_bg4b = pd.read_hdf("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
df_signal = pd.read_hdf("../events/MG3/dataframes/HH4b_picoAOD.h5")
df_3b["signal"] = False
df_bg4b["signal"] = False
df_signal["signal"] = True
raw_df_list = [df_3b, df_bg4b, df_signal]
print("Dataframes loaded")


seeds = np.arange(50)
nbins_list = [2**i for i in range(7)]
experiment_name = "CR_fvt_training_repr_norm"
bins_mode = "quantile"
n_3b = 100_0000

signal_ratios = [0.0, 0.005, 0.0075, 0.01, 0.02]

slope_min = 0.0
slope_max = 0.0
intercept_min = -np.inf
intercept_max = np.inf

test_info_dict_name = f"./data/tmp/test_info_by_hashes.pkl"


if os.path.exists(test_info_dict_name):
    with open(test_info_dict_name, "rb") as f:
        test_info_dict = pickle.load(f)
else:
    test_info_dict = {}


def process_hash(hash):
    corrections, hists = correct_systematic_error(
        hash,
        nbins_list,
        bins_mode,
        intercept_min=intercept_min,
        intercept_max=intercept_max,
        slope_min=slope_min,
        slope_max=slope_max,
        raw_df_list=raw_df_list,
    )

    pulls = {}

    for nbins in hists:
        hist = hists[nbins]
        hist_3b_corrected = hist["3b_corrected"]
        hist_3b_corrected_sq = hist["3b_corrected_sq"]
        hist_4b = hist["4b"]
        hist_4b_sq = hist["4b_sq"]
        hist_diff = hist_4b - hist_3b_corrected
        hist_var = hist_3b_corrected_sq + hist_4b_sq
        pull = hist_diff / np.sqrt(hist_var)
        pulls[nbins] = pull

    return {"corrections": corrections, "hists": hists, "pulls": pulls}


print("Finding hashes to process")
hashes = TrainingInfo.find(
    {
        "experiment_name": experiment_name,
        "model": "FvTClassifier",
        "aux_info_step": 3,
        "dataset": lambda x: x["n_3b"] == n_3b and x["signal_ratio"] in signal_ratios,
    }
)
target_hashes = set(hashes) - set(test_info_dict.keys())
print(f"Number of hashes to process: {len(target_hashes)}")

print("Processing hashes starting")
n_processes = 5
with ProcessPoolExecutor(max_workers=n_processes) as executor:
    test_infos = list(
        tqdm.tqdm(executor.map(process_hash, target_hashes), total=len(target_hashes))
    )
    for hash, test_info in zip(target_hashes, test_infos):
        test_info_dict[hash] = test_info

print("Processing hashes done")


with open(test_info_dict_name, "wb") as f:
    pickle.dump(test_info_dict, f)
