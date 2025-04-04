import pandas as pd
from dataset import MotherSamples
from training_info import TrainingInfo
from utils import select_random_true_elements
from events_data import EventsData, get_is_signal
from signal_region import compute_sr_stats, get_SR_CR_cut
from pathlib import Path


@profile
def main():
    path_3b = Path("../events/MG3/dataframes/threeTag_picoAOD.h5")
    path_bg4b = Path("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
    path_signal = Path("../events/MG3/dataframes/HH4b_picoAOD.h5")
    df_3b = pd.read_hdf(path_3b)
    df_bg4b = pd.read_hdf(path_bg4b)
    df_signal = pd.read_hdf(path_signal)
    loaded_df = {
        path_3b: df_3b,
        path_bg4b: df_bg4b,
        path_signal: df_signal,
    }

    experiment_name = "CR_fvt_training_ensemble_max_smeared"
    n_3b = 100_0000
    ratio_4b = 0.5
    signal_filename = "HH4b_picoAOD.h5"
    SR_size = 0.2
    CR_size = 0.8

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

    hparams_filter = {
        "experiment_name": experiment_name,
        "aux_info_step": 3,
        "dataset": lambda x: (
            x["n_3b"] == n_3b
            and x["ratio_4b"] == ratio_4b
            and x["signal_filename"] == signal_filename
        ),
        "signal_region": lambda x: (
            x["4b_in_SR"] == SR_size and x["4b_in_CR"] == CR_size
        ),
    }
    CR_fvt_hashes = TrainingInfo.find(hparams_filter)
    print(len(CR_fvt_hashes))

    for CR_fvt_hash in CR_fvt_hashes[:2]:
        hparams_filter = {"CR_fvt_hash": CR_fvt_hash}
        mi_test_hash = TrainingInfo.find(hparams_filter)
        assert len(mi_test_hash) == 1, f"len(mi_test_hash) == {len(mi_test_hash)}"
        mi_test_hash = mi_test_hash[0]
        mi_test_tinfo = TrainingInfo.load(mi_test_hash)
        CR_fvt_tinfo = TrainingInfo.load(CR_fvt_hash)
        SR_stats_hashes = CR_fvt_tinfo.hparams["signal_region"]["SR_stats_hashes"]
        ensemble_mode = CR_fvt_tinfo.hparams["signal_region"]["ensemble_mode"]
        stats_type = CR_fvt_tinfo.hparams["signal_region"]["stats_type"]
        signal_filename = CR_fvt_tinfo.hparams["dataset"]["signal_filename"]

        mi_test_test_ratio = mi_test_tinfo.hparams["mi_test_dataset"]["test_ratio"]
        mi_test_data_seed = mi_test_tinfo.hparams["mi_test_dataset"]["data_seed"]

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


if __name__ == "__main__":
    main()
