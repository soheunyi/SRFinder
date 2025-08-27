import numpy as np
from itertools import combinations
import pathlib
import pandas as pd

data_directory = pathlib.Path("../events/MG3")

df_zh4b = pd.read_hdf(data_directory / "dataframes" / "ZH4b_picoAOD.h5")
df_zz4b = pd.read_hdf(data_directory / "dataframes" / "ZZ4b_picoAOD.h5")

# clean signals that have duplicate jets
pt = df_zh4b[["Jet0_pt", "Jet1_pt", "Jet2_pt", "Jet3_pt"]].values
idx = np.zeros(len(pt), dtype=bool)
for i, j in combinations(range(pt.shape[1]), 2):
    idx = idx | (pt[:, i] == pt[:, j])

df_zh4b_cleaned = df_zh4b[~idx].reset_index(drop=True)
df_zh4b_cleaned.sample(frac=1, random_state=0).reset_index(drop=True)
df_zh4b_cleaned.to_hdf(
    data_directory / "dataframes" / "ZH4b_picoAOD_cleaned.h5", key="df", mode="w"
)

df_zz4b_cleaned = df_zz4b[~idx].reset_index(drop=True)
df_zz4b_cleaned.sample(frac=1, random_state=0).reset_index(drop=True)
df_zz4b_cleaned.to_hdf(
    data_directory / "dataframes" / "ZZ4b_picoAOD_cleaned.h5", key="df", mode="w"
)
