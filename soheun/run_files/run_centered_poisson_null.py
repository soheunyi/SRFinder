"""Run the reduced Phase-1 centered-Poisson no-correction KS campaign."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import time

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
DATA_REPO = Path(
    os.environ.get("SRFINDER_DATA_REPO", "/home/export/soheuny/SRFinder/soheun")
)
OUT = DATA_REPO / "data/refit_bootstrap/centered_poisson_null_v1"
MANIFEST = OUT / "manifest.pkl"
BOOTSTRAPS = 1000
ALPHA = 0.05
SEED = 1729
STATISTIC_CLIP = 10.0
VERSION = "centered-poisson1-weighted-ks-v2-cpp"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_data_paths():
    """Point worktree code at the established artifact store."""

    from dataset import MotherSamples
    from training_info import TrainingInfo

    TrainingInfo.SAVE_DIR = DATA_REPO / "data/TrainingInfo"
    TrainingInfo.META_DIR = DATA_REPO / "data/metadata/TrainingInfo.pkl"
    MotherSamples.SAVE_DIR = DATA_REPO / "data/MotherSamples"
    MotherSamples.META_DIR = DATA_REPO / "data/metadata/MotherSamples.pkl"
    return TrainingInfo, MotherSamples


def load_arrays(row: dict):
    from constants import FEATURES
    from events_data import events_from_scdinfo
    from signal_region import compute_sr_stats, get_SR_CR_cut

    TrainingInfo, MotherSamples = configure_data_paths()
    tinfo = TrainingInfo.load(row["hash"])
    cfg = tinfo.hparams["signal_region"]
    dataset = tinfo.hparams["dataset"]
    observed = {
        "experiment_name": tinfo.hparams["experiment_name"],
        "signal_ratio": dataset["signal_ratio"],
        "sr_size": cfg["4b_in_SR"],
        "seed": dataset["seed"],
        "stats_type": cfg["stats_type"],
    }
    for key, value in observed.items():
        if value != row[key]:
            raise RuntimeError(
                f"{row['hash']}: manifest {key}={row[key]} but metadata has {value}"
            )
    if cfg["ensemble_mode"] != "max":
        raise RuntimeError(f"{row['hash']}: expected ensemble maximum")

    signal_filename = dataset["signal_filename"]
    stats_train, stats_test = compute_sr_stats(
        cfg["SR_stats_hashes"],
        signal_filename,
        cfg["ensemble_mode"],
        cfg["stats_type"],
    )
    source = TrainingInfo.load(cfg["SR_stats_hashes"][0])
    if row["stats_type"] == "smeared":
        observed_eta = source.hparams["smearing"]["noise_scale"]
        if observed_eta != row["noise_scale"]:
            raise RuntimeError(
                f"{row['hash']}: manifest eta={row['noise_scale']} but source has "
                f"{observed_eta}"
            )
    elif not np.isinf(row["noise_scale"]):
        raise RuntimeError(f"{row['hash']}: raw FvT row is not eta=infinity")

    mother = MotherSamples.load(source.ms_hash)
    events_train = events_from_scdinfo(
        mother.scdinfo[source.ms_idx], FEATURES, signal_filename
    )
    events_test = events_from_scdinfo(
        mother.scdinfo[~source.ms_idx], FEATURES, signal_filename
    )
    sr_cut, _ = get_SR_CR_cut(
        stats_train,
        events_train,
        {"4b_in_SR": cfg["4b_in_SR"], "4b_in_CR": cfg["4b_in_CR"]},
    )
    in_sr = stats_test >= sr_cut
    scores = stats_test[in_sr]
    events = events_test[in_sr]
    classifier_scores = tinfo.aux_info["fvt_scores_tst_SR"]
    if len(classifier_scores) != len(scores):
        raise RuntimeError(
            f"{row['hash']}: {len(classifier_scores)=} does not match {len(scores)=}"
        )
    weights = np.where(
        events.is_4b,
        events.weights,
        classifier_scores / (1 - classifier_scores) * events.weights,
    )
    scores_3 = scores[~events.is_4b]
    scores_4 = scores[events.is_4b]
    weights_3 = weights[~events.is_4b]
    weights_4 = weights[events.is_4b]
    if np.any(weights_3 < 0) or np.any(weights_4 < 0):
        raise RuntimeError(f"{row['hash']}: Phase-1 nonnegative test has signed weights")
    if np.any(~np.isfinite(scores_3)) or np.any(~np.isfinite(scores_4)):
        raise RuntimeError(f"{row['hash']}: nonfinite SR score")
    clipped_3 = int(np.count_nonzero(np.abs(scores_3) > STATISTIC_CLIP))
    clipped_4 = int(np.count_nonzero(np.abs(scores_4) > STATISTIC_CLIP))
    return (
        np.clip(scores_3, -STATISTIC_CLIP, STATISTIC_CLIP),
        weights_3,
        np.clip(scores_4, -STATISTIC_CLIP, STATISTIC_CLIP),
        weights_4,
        clipped_3,
        clipped_4,
    )


def run_one(row: dict) -> dict:
    from poisson_multiplier_ks import weighted_ks_test

    started = time.perf_counter()
    scores_3, weights_3, scores_4, weights_4, clipped_3, clipped_4 = load_arrays(
        row
    )
    loaded = time.perf_counter()
    linearized = weighted_ks_test(
        scores_3,
        weights_3,
        scores_4,
        weights_4,
        bootstrap_replicates=BOOTSTRAPS,
        alpha=ALPHA,
        seed=SEED,
        form="linearized",
        implementation="cpp",
    )
    direct = weighted_ks_test(
        scores_3,
        weights_3,
        scores_4,
        weights_4,
        bootstrap_replicates=BOOTSTRAPS,
        alpha=ALPHA,
        seed=SEED,
        form="direct_normalized",
        implementation="cpp",
    )
    output = {
        **row,
        "version": VERSION,
        "bootstrap_scheme": "independent_centered_poisson1_multiplier",
        "bootstrap_version": 2,
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "alpha": ALPHA,
        "correction_mode": "none",
        "test_statistic": "weighted_ks_supremum",
        "statistic_clip": STATISTIC_CLIP,
        "n_clipped_3b": clipped_3,
        "n_clipped_4b": clipped_4,
        "linearized": asdict(linearized),
        "direct_normalized": asdict(direct),
        "absolute_p_difference": abs(linearized.p_value - direct.p_value),
        "load_seconds": loaded - started,
        "bootstrap_seconds": time.perf_counter() - loaded,
        "module_sha256": file_sha256(CODE_ROOT / "poisson_multiplier_ks.py"),
        "adapter_sha256": file_sha256(
            CODE_ROOT / "run_files/poisson_multiplier_ks_compiled.py"
        ),
        "kernel_sha256": file_sha256(
            CODE_ROOT / "run_files/poisson_multiplier_ks_kernel.cpp"
        ),
        "runner_sha256": file_sha256(Path(__file__)),
    }
    return output


def validate_checkpoint(result: dict, row: dict) -> None:
    if result.get("version") != VERSION:
        raise RuntimeError(f"{row['hash']}: incompatible checkpoint version")
    if result.get("bootstrap_replicates") != BOOTSTRAPS:
        raise RuntimeError(f"{row['hash']}: incompatible replicate count")
    for key in row:
        observed = result.get(key)
        expected = row[key]
        if isinstance(expected, float) and np.isnan(expected):
            matches = isinstance(observed, float) and np.isnan(observed)
        else:
            matches = observed == expected
        if not matches:
            raise RuntimeError(f"{row['hash']}: checkpoint {key} mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new", type=int)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.n_shards:
        raise ValueError("Require 0 <= shard-index < n-shards")

    # MotherSamples stores event paths relative to the established data
    # checkout.  Isolated code worktrees should still resolve those paths from
    # DATA_REPO rather than copying or symlinking the large HDF5 inputs.
    os.chdir(DATA_REPO)

    with MANIFEST.open("rb") as handle:
        rows = pickle.load(handle)
    selected = rows[args.shard_index :: args.n_shards]
    if args.limit is not None:
        selected = selected[: args.limit]
    print(
        json.dumps(
            {
                "shard_index": args.shard_index,
                "n_shards": args.n_shards,
                "selected": len(selected),
                "campaign_total": len(rows),
            }
        ),
        flush=True,
    )
    new_completed = 0
    for position, row in enumerate(selected, start=1):
        destination = OUT / "results" / f"{row['hash']}.pkl"
        if destination.exists():
            with destination.open("rb") as handle:
                validate_checkpoint(pickle.load(handle), row)
            print(f"skip {position}/{len(selected)} {row['hash']}", flush=True)
            continue
        result = run_one(row)
        temporary = destination.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(result, handle)
        os.replace(temporary, destination)
        new_completed += 1
        print(
            json.dumps(
                {
                    "hash": result["hash"],
                    "noise_scale": result["noise_scale"],
                    "sr_size": result["sr_size"],
                    "seed": result["seed"],
                    "linearized_p": result["linearized"]["p_value"],
                    "direct_p": result["direct_normalized"]["p_value"],
                    "bootstrap_seconds": result["bootstrap_seconds"],
                }
            ),
            flush=True,
        )
        if args.max_new is not None and new_completed >= args.max_new:
            break


if __name__ == "__main__":
    main()
