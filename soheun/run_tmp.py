import numpy as np
import torch
from dataset import MotherSamples
from fvt_classifier import FvTClassifier
from training_info import TrainingInfo
from utils import select_random_true_elements
from events_data import EventsData, get_is_signal
from signal_region import compute_sr_stats, get_SR_CR_cut
import tqdm

n_3b = 100_0000
ratio_4b = 0.5
signal_filename = "HH4b_picoAOD.h5"
experiment_name = "mi_test"

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


mi_test_hashes = TrainingInfo.find({"experiment_name": experiment_name})


for mi_test_hash in tqdm.tqdm(mi_test_hashes):
    mi_test_tinfo = TrainingInfo.load(mi_test_hash)

    CR_fvt_hash = mi_test_tinfo.hparams["CR_fvt_hash"]
    CR_fvt_tinfo = TrainingInfo.load(CR_fvt_hash)
    SR_stats_hashes = CR_fvt_tinfo.hparams["signal_region"]["SR_stats_hashes"]
    ensemble_mode = CR_fvt_tinfo.hparams["signal_region"]["ensemble_mode"]
    stats_type = CR_fvt_tinfo.hparams["signal_region"]["stats_type"]
    signal_filename = CR_fvt_tinfo.hparams["dataset"]["signal_filename"]

    ms_idx = TrainingInfo.load(SR_stats_hashes[0]).ms_idx
    msamples = MotherSamples.load(CR_fvt_tinfo.ms_hash)
    train_scdinfo = msamples.scdinfo[ms_idx]
    tst_scdinfo = msamples.scdinfo[~ms_idx]

    df_train = train_scdinfo.fetch_data()
    df_train["signal"] = get_is_signal(train_scdinfo, signal_filename)
    events_train = EventsData.from_dataframe(df_train, features)

    df_tst = tst_scdinfo.fetch_data()
    df_tst["signal"] = get_is_signal(tst_scdinfo, signal_filename)
    events_tst = EventsData.from_dataframe(df_tst, features)

    mi_test_test_ratio = mi_test_tinfo.hparams["mi_test_dataset"]["test_ratio"]
    mi_test_data_seed = mi_test_tinfo.hparams["mi_test_dataset"]["data_seed"]

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

    tst_idx_int = np.where(~ms_idx)[0]
    tst_SR_idx_bool = np.zeros_like(ms_idx, dtype=bool)
    tst_SR_idx_bool[tst_idx_int[SR_idx]] = True

    tst_SR_test_idx_bool = np.zeros_like(ms_idx, dtype=bool)
    tst_SR_test_idx_bool[tst_idx_int[SR_test_idx]] = True

    events_tst_SR_test = events_tst[SR_test_idx]
    events_tst_SR_train = events_tst[SR_idx & ~SR_test_idx]
    # mi_test_fvt_model = mi_test_tinfo.load_trained_model("best")
    # mi_test_fvt_model.eval()
    # mi_test_fvt_model: FvTClassifier

    CR_fvt_model = CR_fvt_tinfo.load_trained_model("best")
    CR_fvt_model: FvTClassifier
    CR_fvt_model.eval()

    def reweighting_fn(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        fvt_scores = CR_fvt_model.predict(X).detach().cpu()[:, 1]
        return torch.where(y == 0, fvt_scores / (1 - fvt_scores), 1.0)

    reweights_tst_SR_test = (
        reweighting_fn(events_tst_SR_test.X_torch, events_tst_SR_test.is_4b_torch)
        .detach()
        .cpu()
        .numpy()
    )
    reweights_tst_SR_train = (
        reweighting_fn(events_tst_SR_train.X_torch, events_tst_SR_train.is_4b_torch)
        .detach()
        .cpu()
        .numpy()
    )
    mi_test_tinfo.update_aux_info(
        reweights_tst_SR_test=reweights_tst_SR_test,
        reweights_tst_SR_train=reweights_tst_SR_train,
    )

    mi_test_tinfo.save()
