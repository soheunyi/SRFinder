"""Quantify signed ZH event weights in representative eta=2, SR=0.2 samples."""

import argparse
import pickle
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def targets():
    with (
        REPO / "data/refit_bootstrap/continuous_affine_full_v1/manifest.pkl"
    ).open("rb") as handle:
        rows = pickle.load(handle)
    selected = [
        row
        for row in rows
        if row["experiment_name"] == "CR_fvt_training_ensemble_max_ZH4b"
        and row["noise_scale"] == 2.0
        and row["sr_size"] == 0.2
        and row["seed"] == 0
    ]
    selected.sort(key=lambda row: row["signal_ratio"])
    if len(selected) != 6:
        raise RuntimeError(f"Expected six ZH ratios, found {len(selected)}")
    return selected


def main(index: int) -> None:
    from constants import FEATURES
    from dataset import MotherSamples
    from events_data import events_from_scdinfo
    from signal_region import compute_sr_stats, get_SR_CR_cut
    from training_info import TrainingInfo

    row = targets()[index]
    tinfo = TrainingInfo.load(row["hash"])
    cfg = tinfo.hparams["signal_region"]
    filename = tinfo.hparams["dataset"]["signal_filename"]
    train, test = compute_sr_stats(
        cfg["SR_stats_hashes"], filename, cfg["ensemble_mode"], cfg["stats_type"]
    )
    source = TrainingInfo.load(cfg["SR_stats_hashes"][0])
    mother = MotherSamples.load(source.ms_hash)
    train_events = events_from_scdinfo(
        mother.scdinfo[source.ms_idx], FEATURES, filename
    )
    test_events = events_from_scdinfo(
        mother.scdinfo[~source.ms_idx], FEATURES, filename
    )
    cutoff, _ = get_SR_CR_cut(
        train,
        train_events,
        {"4b_in_SR": cfg["4b_in_SR"], "4b_in_CR": cfg["4b_in_CR"]},
    )
    events = test_events[test >= cutoff]
    for category, mask in (
        ("all_4b", events.is_4b),
        ("background_4b", events.is_4b & ~events.is_signal),
        ("signal", events.is_signal),
    ):
        weights = events.weights[mask]
        negative = weights[weights < 0]
        positive = weights[weights > 0]
        print(
            {
                "hash": row["hash"],
                "epsilon": row["signal_ratio"],
                "category": category,
                "n": len(weights),
                "n_negative": len(negative),
                "negative_fraction": len(negative) / len(weights) if len(weights) else 0.0,
                "negative_weight": float(negative.sum()),
                "positive_weight": float(positive.sum()),
                "negative_absolute_share": float(-negative.sum() / positive.sum()) if positive.sum() else 0.0,
                "net_weight": float(weights.sum()),
                "minimum_weight": float(weights.min()) if len(weights) else None,
            },
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=int)
    args = parser.parse_args()
    main(args.index)
