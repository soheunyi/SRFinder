import datetime
import pathlib
import numpy as np
import pandas as pd
import os

import tqdm
from training_info import TrainingInfo
import pickle
from correct_systematic_error import (
    correct_systematic_error,
    correct_systematic_error_mi_test,
)


# from multiprocessing import Pool, Manager


# configs
order = 2
nbins_list = [2**i for i in range(2, 11)]
bins_mode = "quantile"
# bins_stats_type = "fvt"
bins_stats_type = "mi_test"
experiment_names = [
    # "CR_fvt_training_ensemble_max_fvt",
    # "CR_fvt_training_ensemble_max_smeared",
    "mi_test",
]
n_3b = 100_0000
signal_ratios = [0.0, 0.005, 0.0075, 0.01, 0.02]

TrainingInfo.update_metadata()

print(
    f"Configs: order={order}, nbins_list={nbins_list}, bins_mode={bins_mode}, bins_stats_type={bins_stats_type}, experiment_names={experiment_names}, n_3b={n_3b}, signal_ratios={signal_ratios}"
)

print("Loading dataframes")
path_3b = pathlib.Path("../events/MG3/dataframes/threeTag_picoAOD.h5")
path_bg4b = pathlib.Path("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
path_signal = pathlib.Path("../events/MG3/dataframes/HH4b_picoAOD.h5")
df_3b = pd.read_hdf(path_3b)
df_bg4b = pd.read_hdf(path_bg4b)
df_signal = pd.read_hdf(path_signal)
df_3b["signal"] = False
df_bg4b["signal"] = False
df_signal["signal"] = True
loaded_df = {path_3b: df_3b, path_bg4b: df_bg4b, path_signal: df_signal}
print("Dataframes loaded")

if order == 1:
    intercept_min = -np.inf
    intercept_max = np.inf
    slope_min = 0
    slope_max = 0
elif order == 2:
    intercept_min = -np.inf
    intercept_max = np.inf
    slope_min = -np.inf
    slope_max = np.inf
else:
    raise ValueError(f"Invalid order: {order}")


order_str = f"order_{order}"
bins_stats_type_str = f"{bins_stats_type}"
test_info_dict_name = (
    f"./data/tmp/test_info_by_hashes_{order_str}_{bins_stats_type_str}.pkl"
)

if os.path.exists(test_info_dict_name):
    with open(test_info_dict_name, "rb") as f:
        test_info_dict = pickle.load(f)
else:
    test_info_dict = {}


def process_hash(hash, nbins_to_save):
    if bins_stats_type == "mi_test":
        corrections, hists = correct_systematic_error_mi_test(
            hash,
            nbins_to_save,
            bins_mode,
            correction_order=order,
            loaded_df=loaded_df,
        )
    else:
        corrections, hists = correct_systematic_error(
            hash,
            nbins_to_save,
            bins_mode,
            intercept_min=intercept_min,
            intercept_max=intercept_max,
            slope_min=slope_min,
            slope_max=slope_max,
            loaded_df=loaded_df,
        )

    pulls = {}

    for nbins in nbins_to_save:
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


def process_and_save(hash):
    print(f"[{datetime.datetime.now()}] Processing hash: {hash}")
    existing_nbins = test_info_dict.get(hash, {}).get("hists", {}).keys()
    nbins_to_save = [nbins for nbins in nbins_list if nbins not in existing_nbins]
    if len(nbins_to_save) == 0:
        print(f"[{datetime.datetime.now()}] No new nbins to save for hash: {hash}")
        return
    result = process_hash(hash, nbins_to_save)
    if hash not in test_info_dict:
        test_info_dict[hash] = result
    else:
        for key in result.keys():
            test_info_dict[hash][key].update(result[key])

    with open(test_info_dict_name, "wb") as f:
        pickle.dump(test_info_dict, f)


print("Finding hashes to process")
hashes = TrainingInfo.find(
    {
        "experiment_name": lambda x: x in experiment_names,
        "model": "FvTClassifier",
        "dataset": lambda x: x["n_3b"] == n_3b and x["signal_ratio"] in signal_ratios,
        # "aux_info_step": 3,
        "aux_info_step": 4,
        "resample": lambda x: x is None or not x,
    }
)
# target_hashes = [h for h in hashes if h not in test_info_dict.keys()]
target_hashes = list(hashes)
# sort by hash names
target_hashes.sort()

print(f"Number of hashes to process: {len(target_hashes)}")
print("Processing hashes starting")


for hash in tqdm.tqdm(target_hashes):
    process_and_save(hash)

print("Processing hashes done")
