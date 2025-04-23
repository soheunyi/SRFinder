import numpy as np
from copy import deepcopy
from utils import require_keys
from training_info import TrainingInfo
from dataset import MotherSamples
from events_data import EventsData, events_from_scdinfo
from constants import FEATURES
from signal_region import get_SR_CR_cut, compute_sr_stats


def get_step_3_tinfo_events(
    CR_fvt_hparams: dict,
) -> tuple[TrainingInfo, EventsData, EventsData, np.ndarray[bool], np.ndarray[bool]]:
    require_keys(
        CR_fvt_hparams["signal_region"],
        ["SR_stats_hashes", "ensemble_mode", "stats_type"],
    )
    require_keys(CR_fvt_hparams["dataset"], ["signal_filename"])

    SR_stats_hashes = CR_fvt_hparams["signal_region"]["SR_stats_hashes"]
    ensemble_mode = CR_fvt_hparams["signal_region"]["ensemble_mode"]
    stats_type = CR_fvt_hparams["signal_region"]["stats_type"]
    signal_filename = CR_fvt_hparams["dataset"]["signal_filename"]

    # Use the same mother samples and exclude ones used for training base & smeared FvT model
    tinfo_0 = TrainingInfo.load(SR_stats_hashes[0])
    assert tinfo_0.aux_info["step"] == 2
    for hash in SR_stats_hashes:
        tinfo = TrainingInfo.load(hash)
        assert tinfo.ms_hash == tinfo_0.ms_hash
        assert np.all(tinfo.ms_idx == tinfo_0.ms_idx)
        assert tinfo.aux_info["step"] == 2

    msamples = MotherSamples.load(tinfo_0.ms_hash)
    events_train = events_from_scdinfo(
        msamples.scdinfo[tinfo_0.ms_idx], FEATURES, signal_filename
    )
    events_tst = events_from_scdinfo(
        msamples.scdinfo[~tinfo_0.ms_idx], FEATURES, signal_filename
    )

    SR_stats_train, SR_stats_tst = compute_sr_stats(
        SR_stats_hashes,
        signal_filename,
        ensemble_mode,
        stats_type,
    )

    SR_cut, CR_cut = get_SR_CR_cut(
        SR_stats_train, events_train, CR_fvt_hparams["signal_region"]
    )
    CR_idx = (SR_stats_tst >= CR_cut) & (SR_stats_tst < SR_cut)
    SR_idx = SR_stats_tst >= SR_cut
    SR_idx_train = SR_stats_train >= SR_cut

    tst_ms_idx = ~tinfo_0.ms_idx
    tst_ms_idx_int = tst_ms_idx.nonzero()[0]
    CR_ms_idx_int = tst_ms_idx_int[CR_idx]
    CR_ms_idx_bool = np.zeros_like(tst_ms_idx, dtype=bool)
    CR_ms_idx_bool[CR_ms_idx_int] = True

    CR_fvt_tinfo = TrainingInfo(
        CR_fvt_hparams,
        ms_hash=tinfo_0.ms_hash,
        ms_idx=CR_ms_idx_bool,
    )
    return CR_fvt_tinfo, events_train, events_tst, SR_idx_train, SR_idx


def check_and_get_CR_fvt_hparams(config: dict):
    require_keys(
        config,
        [
            "experiment_name",
            "dataset",
            "signal_region",
            "CR_fvt",
            "previous_step_experiment_name",
        ],
    )
    require_keys(
        config["dataset"],
        [
            "signal_filename",
            "signal_ratio",
            "n_3b",
            "ratio_4b",
            "seed",
        ],
    )
    require_keys(
        config["signal_region"],
        ["4b_in_SR", "4b_in_CR", "ensemble_mode", "stats_type", "SR_stats_hashes"],
    )
    require_keys(
        config["CR_fvt"],
        [
            "model",
            "depth",
            "fit_batch_size",
            "model_seed",
            "train_seed",
            "data_seed",
            "max_epochs",
            "val_ratio",
            "early_stop_patience",
            "optimizer",
            "lr_scheduler",
            "dataloader",
            "encoder_mode",
        ],
    )
    require_keys(config["CR_fvt"]["optimizer"], ["type", "lr"])
    require_keys(
        config["CR_fvt"]["lr_scheduler"],
        ["type"],
    )
    if config["CR_fvt"]["lr_scheduler"]["type"] == "ReduceLROnPlateau":
        require_keys(
            config["CR_fvt"]["lr_scheduler"],
            ["factor", "threshold", "patience", "cooldown", "min_lr"],
        )
    if config["CR_fvt"]["optimizer"]["type"] == "AdamWScheduleFree":
        require_keys(
            config["CR_fvt"]["optimizer"],
            ["warmup_steps"],
        )
    require_keys(
        config["CR_fvt"]["dataloader"],
        ["batch_size", "batch_size_multiplier", "batch_size_milestones"],
    )

    if config["CR_fvt"]["model"] == "FvTClassifier":
        require_keys(
            config["CR_fvt"],
            ["dim_dijet_features", "dim_quadjet_features"],
        )
        assert isinstance(config["CR_fvt"]["depth"], dict), "depth must be a dictionary"
        require_keys(
            config["CR_fvt"]["depth"],
            ["encoder", "decoder"],
        )
    elif config["CR_fvt"]["model"] == "AttentionClassifier":
        require_keys(
            config["CR_fvt"],
            ["dim_q"],
        )

    CR_fvt_hparams = deepcopy(config["CR_fvt"])
    CR_fvt_hparams["experiment_name"] = config["experiment_name"]
    CR_fvt_hparams["dataset"] = config["dataset"]
    CR_fvt_hparams["signal_region"] = config["signal_region"]
    CR_fvt_hparams["step"] = 3

    return CR_fvt_hparams
