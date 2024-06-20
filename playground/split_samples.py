import pathlib
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append("/home/soheuny/HH4bsim")

data_directory = pathlib.Path("/home/soheuny/HH4bsim/events/MG3")

df3B = pd.read_hdf(data_directory / "dataframes" / "symmetrized_bbbj.h5")
df4B = pd.read_hdf(data_directory / "dataframes" / "symmetrized_bbbb_large.h5")
dfHH4B = pd.read_hdf(data_directory / "dataframes" / "symmetrized_HH4b.h5")

# seed
np.random.seed(42)

DR_RATIO = 0.5  # ratio of data used for density ratio estimation
AE_RATIO = 0.4  # ratio of data used for autoencoder training
TEST_RATIO = 0.1

assert DR_RATIO + AE_RATIO + TEST_RATIO == 1

# DR_RATIO of df3B, df4B for density ratio estimation
rand_perm_3B = np.random.permutation(df3B.index.size)
DR_3B_INDEX = rand_perm_3B[: int(DR_RATIO * df3B.index.size)]

rand_perm_4B = np.random.permutation(df4B.index.size)
DR_4B_INDEX = rand_perm_4B[: int(DR_RATIO * df4B.index.size)]

# Other half for training autoencoder
AE_3B_INDEX = rand_perm_3B[
    int(DR_RATIO * df3B.index.size) : int((DR_RATIO + AE_RATIO) * df3B.index.size)
]
AE_4B_INDEX = rand_perm_4B[
    int(DR_RATIO * df4B.index.size) : int((DR_RATIO + AE_RATIO) * df4B.index.size)
]

# Rest for testing
TEST_3B_INDEX = rand_perm_3B[int((DR_RATIO + AE_RATIO) * df3B.index.size) :]
TEST_4B_INDEX = rand_perm_4B[int((DR_RATIO + AE_RATIO) * df4B.index.size) :]


def split_samples(*data, at: float, div: int = None, seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    for d in data:
        assert len(d) == len(data[0])

    if div is None:
        split_at = int(at * len(data[0]))
        end_at = len(data[0])
    else:
        split_at = div * (int(at * len(data[0])) // div)
        end_at = div * (len(data[0]) // div)

    # shuffle
    indices = np.random.permutation(len(data[0]))
    data = [d[indices] for d in data]

    return [d[:split_at] for d in data], [d[split_at:end_at] for d in data]
