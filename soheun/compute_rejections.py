import time

import click
import pandas as pd

start_time = time.time()

from itertools import product
from scipy import stats
from training_info import TrainingInfo
import numpy as np
import pickle

# order = 2
# experiment_name = "CR_fvt_training_ensemble_max_smeared"


@click.command()
@click.option("--order", type=int, required=True, help="Order of correction")
@click.option(
    "--experiment-name",
    type=str,
    required=True,
    help="Name of the experiment",
)
@click.option("--sr-size", type=float, required=True, help="Size of the signal region")
@click.option("--n-3b", type=int, required=True, help="Number of 3b", default=100_0000)
@click.option(
    "--nbins-list",
    type=list,
    required=True,
    help="Number of bins",
    default=[2**i for i in range(2, 9)],
)
@click.option(
    "--signal-ratios",
    type=list,
    required=True,
    help="Signal ratios",
    default=[0.0, 0.005, 0.0075, 0.01, 0.02],
)
@click.option(
    "--sig-level",
    type=float,
    required=True,
    help="Significance level",
    default=0.05,
)
def main(order, experiment_name, sr_size, n_3b, nbins_list, signal_ratios, sig_level):
    compute_rejections(
        order, experiment_name, sr_size, n_3b, nbins_list, signal_ratios, sig_level
    )


def compute_rejections(
    order,
    experiment_name,
    sr_size,
    n_3b,
    nbins_list,
    signal_ratios,
    sig_level,
    bins_stats_type,
    verbose=True,
):
    if order not in [0, 1, 2]:
        raise ValueError(f"Unsupported order: {order}")
    if bins_stats_type not in ["sr_stats", "fvt"]:
        raise ValueError(f"Unsupported bins_stats_type: {bins_stats_type}")

    order_str = f"order_{order}"
    bins_stats_type_str = f"{bins_stats_type}"
    test_info_dict_name = (
        f"./data/tmp/test_info_by_hashes_{order_str}_{bins_stats_type_str}.pkl"
    )
    # print(f"Loading test info dict, time spend={time.time()-start_time}")
    with open(test_info_dict_name, "rb") as f:
        test_info_dict = pickle.load(f)

    pull_arrays = {
        (signal_ratio, nbins): []
        for signal_ratio, nbins in product(signal_ratios, nbins_list)
    }

    # print(f"Experiment name: {experiment_name}")
    hashes = TrainingInfo.find(
        {
            "experiment_name": experiment_name,
            "dataset": lambda x: x["n_3b"] == n_3b
            and x["signal_ratio"] in signal_ratios,
            "signal_region": lambda x: x["4b_in_SR"] == sr_size,
        }
    )
    # print(f"Enumerating hashes, time spend={time.time()-start_time}")
    n_hashes = 0
    for hash in hashes:
        tinfo = TrainingInfo.load(hash)
        signal_ratio = tinfo.hparams["dataset"]["signal_ratio"]
        pulls = test_info_dict[hash]["pulls"]
        for nbins in nbins_list:
            pull_arrays[(signal_ratio, nbins)].append(pulls[nbins])
        n_hashes += 1

    # print(f"Number of hashes: {n_hashes}")
    pull_arrays = {k: np.array(v) for k, v in pull_arrays.items()}

    rej_df_list = []

    if verbose:
        print(f"Correction DF = {order}")
    for nbins, signal_ratio in product(nbins_list, signal_ratios):
        if verbose:
            print(f"signal_ratio = {signal_ratio}, nbins = {nbins}")
        chi2_stat = np.sqrt(np.mean(pull_arrays[(signal_ratio, nbins)] ** 2, axis=1))
        z_rej = stats.chi2.ppf(1 - sig_level, df=nbins - 1)
        z_rej = np.sqrt(z_rej / nbins)
        if verbose:
            print(np.mean(chi2_stat > z_rej))
        rej_df_list.append(
            {
                "nbins": nbins,
                "signal_ratio": signal_ratio,
                "rej": np.mean(chi2_stat > z_rej),
            }
        )

    rej_df = pd.DataFrame(rej_df_list)
    return rej_df


if __name__ == "__main__":
    main()
