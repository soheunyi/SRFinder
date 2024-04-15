import ROOT
from array import array
import sys
import pathlib

sys.path.insert(0, "../..")
sys.path.insert(0, "../")

from model_train import modelParameters
from symmetrized_model_train import symmetrizedModelParameters
from symmetrize_df import symmetrize_df
import pandas as pd
import numpy as np

pathlib.Path("fvt_fit").mkdir(parents=True, exist_ok=True)

df3b = pd.read_hdf("../../../events/MG3/dataframes/bbbj.h5")
df4b = pd.read_hdf("../../../events/MG3/dataframes/bbbb_large.h5")

df3b = symmetrize_df(df3b)
df4b = symmetrize_df(df4b)

# df3b = df3b[df3b['CR']]
# df4b = df4b[df4b['CR']]

# df4b.weight /= df4b.weight  # force all 4b weights to 1
# df3b.weight /= df3b.weight  # force all 4b weights to 1 # modified
# df3b.weight *= df4b[df4b['CR']].shape[0]/df3b.weight.sum() # normalize 3b sample in CR
# df3b.weight *= df4b.shape[0] / df3b.weight.sum()  # normalize 3b sample in CR

# model = modelParameters(df3b[df3b['CR']], df4b[df4b['CR']], model_path="fvt_fit/", classifier="FvT")
# model = modelParameters(df3b, df4b, model_path="fvt_fit/", classifier="FvT", epochs=100)
model = symmetrizedModelParameters(
    df3b,
    df4b,
    model_path="fvt_fit/",
    classifier="FvT",
    epochs=10,
    outputName="",
)
model.trainSetup()
model.runEpochs(print_all_epochs=True)
