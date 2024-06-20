import pathlib
import sys

import pandas as pd

sys.path.append("/home/soheuny/HH4bsim")
from python.classifier.symmetrize_df import symmetrize_df


data_directory = pathlib.Path("/home/soheuny/HH4bsim/events/MG3")

df_3b = pd.read_hdf(data_directory / "dataframes" / "threeTag_picoAOD.h5")
df_3b = symmetrize_df(df_3b)
df_3b["fourTag"] = False
df_3b.to_hdf(data_directory / "dataframes" / "threeTag_picoAOD.h5", key="df", mode="w")

df_bg4b = pd.read_hdf(data_directory / "dataframes" / "fourTag_picoAOD.h5")
df_bg4b = symmetrize_df(df_bg4b)
df_bg4b["fourTag"] = True
df_bg4b.to_hdf(data_directory / "dataframes" / "fourTag_picoAOD.h5", key="df", mode="w")

df_bg4b_large = pd.read_hdf(data_directory / "dataframes" / "fourTag_10x_picoAOD.h5")
df_bg4b_large = symmetrize_df(df_bg4b_large)
df_bg4b_large["fourTag"] = True
df_bg4b_large.to_hdf(
    data_directory / "dataframes" / "fourTag_10x_picoAOD.h5", key="df", mode="w"
)

df_hh4b = pd.read_hdf(data_directory / "dataframes" / "HH4b_picoAOD.h5")
df_hh4b = symmetrize_df(df_hh4b)
df_hh4b["fourTag"] = True
df_hh4b.to_hdf(data_directory / "dataframes" / "HH4b_picoAOD.h5", key="df", mode="w")

# df_3b = pd.read_hdf(data_directory / "dataframes" / "bbbj.h5")
# df_3b = symmetrize_df(df_3b)
# df_3b.to_hdf(data_directory / "dataframes" / "symmetrized_bbbj.h5", key="df", mode="w")

# df_bg4b = pd.read_hdf(data_directory / "dataframes" / "bbbb_large.h5")
# df_bg4b = symmetrize_df(df_bg4b)
# df_bg4b.to_hdf(
#     data_directory / "dataframes" / "symmetrized_bbbb_large.h5", key="df", mode="w"
# )

# df_hh4b = pd.read_hdf(data_directory / "dataframes" / "HH4b.h5")
# df_hh4b = symmetrize_df(df_hh4b)
# df_hh4b.to_hdf(data_directory / "dataframes" / "symmetrized_HH4b.h5", key="df", mode="w")
