import time

start_time = time.time()

from itertools import product
from scipy import stats
from training_info import TrainingInfo
import numpy as np
import pickle

sig_level = 0.05
nbins_list = [2**i for i in range(1, 7)]
signal_ratios = [0.0, 0.005, 0.0075, 0.01, 0.02]

test_info_dict_name = f"./data/tmp/test_info_by_hashes.pkl"

print(f"Loading test info dict, time spend={time.time()-start_time}")
with open(test_info_dict_name, "rb") as f:
    test_info_dict = pickle.load(f)

pull_arrays = {
    (signal_ratio, nbins): []
    for signal_ratio, nbins in product(signal_ratios, nbins_list)
}

experiment_name = "CR_fvt_training_ensemble_max_fvt"
print(f"Experiment name: {experiment_name}")
print(f"Enumerating hashes, time spend={time.time()-start_time}")
n_hashes = 0
for hash in test_info_dict:
    tinfo = TrainingInfo.load(hash)
    if tinfo.hparams["experiment_name"] != experiment_name:
        continue
    seed = tinfo.hparams["dataset"]["seed"]
    signal_ratio = tinfo.hparams["dataset"]["signal_ratio"]
    pulls = test_info_dict[hash]["pulls"]
    for nbins in nbins_list:
        pull_arrays[(signal_ratio, nbins)].append(pulls[nbins])
    n_hashes += 1

print(f"Number of hashes: {n_hashes}")
pull_arrays = {k: np.array(v) for k, v in pull_arrays.items()}

for nbins, signal_ratio in product(nbins_list, signal_ratios):
    print(f"signal_ratio = {signal_ratio}, nbins = {nbins}")
    chi2_stat = np.sqrt(np.mean(pull_arrays[(signal_ratio, nbins)] ** 2, axis=1))
    z_rej = stats.chi2.ppf(1 - sig_level, df=nbins - 1)
    z_rej = np.sqrt(z_rej / nbins)
    print(np.mean(chi2_stat > z_rej))
