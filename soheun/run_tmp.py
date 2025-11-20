from training_info import TrainingInfo
import pandas as pd

experiment_name = "smeared_fvt_training_ensemble_ZH4b"
TrainingInfo.update_metadata()
metadata = TrainingInfo.load_metadata()

hashes = TrainingInfo.find({"experiment_name": experiment_name})
hparams_df = pd.DataFrame([
    {"hash": hash_,
    "seed": metadata[hash_]["dataset"]["seed"],
    "signal_ratio": metadata[hash_]["dataset"]["signal_ratio"],
    "n_3b": metadata[hash_]["dataset"]["n_3b"],
    "ratio_4b": metadata[hash_]["dataset"]["ratio_4b"],
    "signal_filename": metadata[hash_]["dataset"]["signal_filename"],
    "train_seed": metadata[hash_]["train_seed"],
    "noise_scale": metadata[hash_]["smearing"]["noise_scale"],
    },
    for hash_ in hashes
   ])