import pathlib
import sys

import pandas as pd

sys.path.append("/home/soheuny/HH4bsim")
from python.classifier.symmetrize_df import symmetrize_df


data_directory = pathlib.Path("/home/soheuny/HH4bsim/events/MG3")

df3B = pd.read_hdf(data_directory / "dataframes" / "bbbj.h5")
df3B = symmetrize_df(df3B)
df3B.to_hdf(data_directory / "dataframes" / "symmetrized_bbbj.h5", key="df", mode="w")

df4B = pd.read_hdf(data_directory / "dataframes" / "bbbb_large.h5")
df4B = symmetrize_df(df4B)
df4B.to_hdf(
    data_directory / "dataframes" / "symmetrized_bbbb_large.h5", key="df", mode="w"
)

dfHH4B = pd.read_hdf(data_directory / "dataframes" / "HH4b.h5")
dfHH4B = symmetrize_df(dfHH4B)
dfHH4B.to_hdf(data_directory / "dataframes" / "symmetrized_HH4b.h5", key="df", mode="w")
