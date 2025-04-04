from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import tqdm


from signal_region import get_SR_CR_cut, compute_sr_stats
from events_data import EventsData, get_is_signal
from training_info import TrainingInfo
from utils import select_random_true_elements
from mi_test import mi_test
from dataset import MotherSamples


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

calibrator = None
if calibrator == "platt":
    save_filename = "./data/tmp/calibrated_mi_test_p_values_dict.pkl"
else:
    save_filename = "./data/tmp/mi_test_p_values_dict.pkl"


def calibrate_fvt_scores_cv(
    fvt_scores: np.ndarray,
    is_4b: np.ndarray,
    weights: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
    calibrator: str = "isotonic",
) -> np.ndarray:
    """
    Calibrate classifier scores using Platt scaling (logistic regression) with cross-validation.

    Args:
        fvt_scores: Raw classifier scores (shape: [n_samples]).
        is_4b: Binary labels (0 or 1, shape: [n_samples]).
        weights: Sample weights (shape: [n_samples]).
        n_folds: Number of cross-validation folds.
        random_state: Random seed for reproducibility.

    Returns:
        Calibrated probabilities (shape: [n_samples]).
    """
    if calibrator is None:
        return fvt_scores
    calibrated_probs = np.zeros_like(fvt_scores)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    y_min = np.min(fvt_scores)
    y_max = np.max(fvt_scores)

    for train_idx, test_idx in skf.split(fvt_scores, is_4b):
        # Reshape scores to 2D (required by scikit-learn)
        scores_train = fvt_scores[train_idx].reshape(-1, 1)
        scores_test = fvt_scores[test_idx].reshape(-1, 1)

        if calibrator == "isotonic":
            regressor = IsotonicRegression(
                out_of_bounds="clip", y_min=y_min, y_max=y_max
            )
        elif calibrator == "platt":
            # Fit logistic regression (Platt scaling)
            regressor = LogisticRegression(
                penalty=None,  # Disable regularization (Platt scaling uses unregularized LR)
                solver="lbfgs",  # Solver for unconstrained optimization
                max_iter=1000,  # Ensure convergence
            )
        else:
            raise ValueError(f"Invalid calibrator: {calibrator}")

        regressor.fit(scores_train, is_4b[train_idx], sample_weight=weights[train_idx])

        # Predict probabilities on the test fold
        calibrated_probs[test_idx] = regressor.predict_proba(scores_test)[:, 1]

    return calibrated_probs


features = [
    "sym_Jet0_pt",
    "sym_Jet1_pt",
    "sym_Jet2_pt",
    "sym_Jet3_pt",
    "sym_Jet0_eta",
    "sym_Jet1_eta",
    "sym_Jet2_eta",
    "sym_Jet3_eta",
    "sym_Jet0_phi",
    "sym_Jet1_phi",
    "sym_Jet2_phi",
    "sym_Jet3_phi",
    "sym_Jet0_m",
    "sym_Jet1_m",
    "sym_Jet2_m",
    "sym_Jet3_m",
]

path_3b = Path("../events/MG3/dataframes/threeTag_picoAOD.h5")
path_4b = Path("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
path_signal = Path("../events/MG3/dataframes/HH4b_picoAOD.h5")
df_3b = pd.read_hdf(path_3b)
df_bg4b = pd.read_hdf(path_4b)
df_signal = pd.read_hdf(path_signal)
df_3b["signal"] = False
df_bg4b["signal"] = False
df_signal["signal"] = True
raw_df_list = [df_3b, df_bg4b, df_signal]
loaded_df = {path_3b: df_3b, path_4b: df_bg4b, path_signal: df_signal}

n_3b = 100_0000
ratio_4b = 0.5
signal_filename = "HH4b_picoAOD.h5"
experiment_name = "CR_fvt_training_ensemble_max_smeared"
N_REPS = 1000
tst_seed = 42

SR_CR_sizes = [
    (0.05, 0.95),
    (0.1, 0.9),
    (0.15, 0.85),
    (0.2, 0.8),
]


signal_ratios = [0.0, 0.005, 0.0075, 0.01, 0.02]
p_values_dict = {sr[0]: {} for sr in SR_CR_sizes}
sig_level = 0.05

for signal_ratio in signal_ratios:
    for SR_size, CR_size in SR_CR_sizes:
        print("signal_ratio", signal_ratio, "SR_CR_size", SR_size, CR_size)
        hparams_filter = {
            "experiment_name": experiment_name,
            "aux_info_step": 3,
            "dataset": lambda x: (
                x["n_3b"] == n_3b
                and x["ratio_4b"] == ratio_4b
                and x["signal_filename"] == signal_filename
                and x["signal_ratio"] == signal_ratio
            ),
            "signal_region": lambda x: (
                x["4b_in_SR"] == SR_size and x["4b_in_CR"] == CR_size
            ),
        }
        CR_fvt_hashes = TrainingInfo.find(hparams_filter)

        p_value_MCE_list = []
        p_value_AUC_list = []
        p_value_RENYI_list = []

        for CR_fvt_hash in tqdm.tqdm(CR_fvt_hashes):
            hparams_filter = {
                "CR_fvt_hash": CR_fvt_hash,
                "dataloader": lambda x: x["batch_size"] == 256,
            }
            mi_test_hash = TrainingInfo.find(hparams_filter)
            if not len(mi_test_hash) == 1:
                for h in mi_test_hash:
                    print(TrainingInfo.load(h).hparams)
                raise ValueError("len(mi_test_hash) != 1")
            mi_test_hash = mi_test_hash[0]
            mi_test_tinfo = TrainingInfo.load(mi_test_hash)
            CR_fvt_tinfo = TrainingInfo.load(CR_fvt_hash)
            SR_stats_hashes = CR_fvt_tinfo.hparams["signal_region"]["SR_stats_hashes"]
            ensemble_mode = CR_fvt_tinfo.hparams["signal_region"]["ensemble_mode"]
            stats_type = CR_fvt_tinfo.hparams["signal_region"]["stats_type"]
            signal_filename = CR_fvt_tinfo.hparams["dataset"]["signal_filename"]

            ms_idx = TrainingInfo.load(SR_stats_hashes[0]).ms_idx
            msamples = MotherSamples.load(CR_fvt_tinfo.ms_hash)
            train_scdinfo = msamples.scdinfo[ms_idx]
            tst_scdinfo = msamples.scdinfo[~ms_idx]

            df_train = train_scdinfo.fetch_data(loaded_df)
            df_train["signal"] = get_is_signal(train_scdinfo, signal_filename)
            events_train = EventsData.from_dataframe(df_train, features)

            df_tst = tst_scdinfo.fetch_data(loaded_df)
            df_tst["signal"] = get_is_signal(tst_scdinfo, signal_filename)
            events_tst = EventsData.from_dataframe(df_tst, features)

            SR_stats_train, SR_stats_tst = compute_sr_stats(
                SR_stats_hashes,
                signal_filename,
                ensemble_mode,
                stats_type,
            )

            SR_cut, _ = get_SR_CR_cut(
                SR_stats_train, events_train, CR_fvt_tinfo.hparams["signal_region"]
            )
            SR_idx = SR_stats_tst >= SR_cut

            SR_3b_idx = events_tst.is_3b & SR_idx
            SR_4b_idx = events_tst.is_4b & SR_idx

            mi_test_test_ratio = mi_test_tinfo.hparams["mi_test_dataset"]["test_ratio"]
            mi_test_data_seed = mi_test_tinfo.hparams["mi_test_dataset"]["data_seed"]

            # select test_ratio of the events in the SR for 3b and 4b
            SR_3b_test_idx = select_random_true_elements(
                SR_3b_idx,
                mi_test_test_ratio,
                mi_test_data_seed,
            )
            SR_4b_test_idx = select_random_true_elements(
                SR_4b_idx,
                mi_test_test_ratio,
                mi_test_data_seed,
            )

            SR_test_idx = SR_3b_test_idx | SR_4b_test_idx
            events_tst_SR_test = events_tst[SR_test_idx]
            events_tst_SR_train = events_tst[SR_idx & ~SR_test_idx]
            fvt_scores_tst_SR_test = mi_test_tinfo.aux_info["fvt_scores_tst_SR_test"]
            reweights_tst_SR_test = mi_test_tinfo.aux_info["reweights_tst_SR_test"]
            reweights_tst_SR_train = mi_test_tinfo.aux_info["reweights_tst_SR_train"]

            SR_test_idx = SR_3b_test_idx | SR_4b_test_idx
            SR_test_idx_in_SR_tst = SR_test_idx[SR_idx]

            tst_idx = np.zeros_like(SR_3b_test_idx, dtype=int)
            tst_idx[SR_3b_test_idx] = 1
            tst_idx[SR_4b_test_idx] = 2
            tst_idx = tst_idx[tst_idx != 0]
            SR_3b_test_idx = tst_idx == 1
            SR_4b_test_idx = tst_idx == 2

            CR_fvt_scores_tst_SR_test = (CR_fvt_tinfo.aux_info["fvt_scores_tst_SR"])[
                SR_test_idx_in_SR_tst
            ]
            CR_reweights_tst_SR_test = np.where(
                events_tst_SR_test.is_4b,
                1,
                CR_fvt_scores_tst_SR_test / (1 - CR_fvt_scores_tst_SR_test),
            )
            events_tst_SR_test_clone = events_tst_SR_test.clone()
            events_tst_SR_test_clone.reweight(
                CR_reweights_tst_SR_test * events_tst_SR_test.weights
            )

            weights_test = events_tst_SR_test.weights * reweights_tst_SR_test
            weights_train = events_tst_SR_train.weights * reweights_tst_SR_train

            fvt_scores_tst_SR_test = calibrate_fvt_scores_cv(
                fvt_scores_tst_SR_test,
                events_tst_SR_test.is_4b,
                weights_test,
                n_folds=5,
                random_state=42,
                calibrator=calibrator,
            )

            SR_3b_train_idx = events_tst_SR_train.is_3b
            SR_4b_train_idx = events_tst_SR_train.is_4b
            w_3b_train = np.sum(weights_train[SR_3b_train_idx])
            n_3b_train = np.sum(SR_3b_train_idx)
            w_4b_train = np.sum(weights_train[SR_4b_train_idx])
            n_4b_train = np.sum(SR_4b_train_idx)

            pi = w_4b_train / (w_3b_train + w_4b_train)

            (
                auc_score_orig,
                mce_score_orig,
                renyi_score_orig,
                auc_score_null,
                mce_score_null,
                renyi_score_null,
            ) = mi_test(
                fvt_scores_tst_SR_test,
                events_tst_SR_test.is_4b,
                weights_test,
                pi,
                "permutation",
                N_REPS,
                tst_seed,
                do_tqdm=False,
            )

            p_value_MCE = np.sum(mce_score_null <= mce_score_orig) / N_REPS
            p_value_AUC = np.sum(auc_score_null >= auc_score_orig) / N_REPS
            p_value_RENYI = np.sum(renyi_score_null >= renyi_score_orig) / N_REPS
            p_value_MCE_list.append(p_value_MCE)
            p_value_AUC_list.append(p_value_AUC)
            p_value_RENYI_list.append(p_value_RENYI)

        p_value_MCE_list = np.array(p_value_MCE_list)
        p_value_AUC_list = np.array(p_value_AUC_list)
        p_value_RENYI_list = np.array(p_value_RENYI_list)
        print(f"SR_size: {SR_size}, signal_ratio: {signal_ratio}")
        print("MCE", np.mean(p_value_MCE_list < sig_level))
        print(p_value_MCE_list)
        print("AUC", np.mean(p_value_AUC_list < sig_level))
        print(p_value_AUC_list)
        print("Renyi", np.mean(p_value_RENYI_list < sig_level))
        print(p_value_RENYI_list)
        p_values_dict[SR_size][signal_ratio] = {
            "MCE": p_value_MCE_list,
            "AUC": p_value_AUC_list,
            "Renyi": p_value_RENYI_list,
        }

with open(save_filename, "wb") as f:
    pickle.dump(p_values_dict, f)
