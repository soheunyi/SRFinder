import sys
import pathlib

import sys

sys.path.append("/home/soheuny/HH4bsim")


from python.classifier.symmetrized_model_train import symmetrizedModelParameters
import pandas as pd
from split_samples import DR_3B_INDEX, DR_4B_INDEX

pathlib.Path("fvt_fit").mkdir(parents=True, exist_ok=True)

data_directory = pathlib.Path("/home/soheuny/HH4bsim/events/MG3")

df3B = pd.read_hdf(data_directory / "dataframes" / "symmetrized_bbbj.h5")
df4B = pd.read_hdf(data_directory / "dataframes" / "symmetrized_bbbb_large.h5")

model = symmetrizedModelParameters(
    df3B.iloc[DR_3B_INDEX],
    df4B.iloc[DR_4B_INDEX],
    model_path="fvt_fit/",
    classifier="FvT",
    epochs=30,
    outputName="",
)
model.trainSetup()
model.runEpochs(print_all_epochs=True)
