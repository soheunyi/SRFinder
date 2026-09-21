import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/export/soheuny/SRFinder/soheun")
sys.path.insert(0, str(REPO))

from constants import FEATURES
from dataset import MotherSamples
from events_data import events_from_scdinfo
from ks_test import affine_tilt, max_cdf_diff
from signal_region import compute_sr_stats, get_SR_CR_cut
from training_info import TrainingInfo


OUTPUT_DIR = REPO / "data/refit_bootstrap/null_eta2_logitcap10_v1"
GRID_SIZE = 0.005
CDF_MODE = "mean"
N_REPS = 1000
STATISTIC_CLIP = 10.0


def affine_correction_fast(stats_3b, stats_4b, weights_3b, weights_4b):
    mean_3b = np.sum(weights_3b * stats_3b) / np.sum(weights_3b)
    std_3b = np.sqrt(
        np.sum(weights_3b * (stats_3b - mean_3b) ** 2) / np.sum(weights_3b)
    )
    centered = (stats_3b - mean_3b) / std_3b
    slope_max = -1 / np.min(centered)
    slope_min = -1 / np.max(centered)
    grid = GRID_SIZE * np.arange(
        int(slope_min / GRID_SIZE), int(slope_max / GRID_SIZE) + 1
    )

    order = np.argsort(np.concatenate([stats_3b, stats_4b]))
    zeros_3b = np.zeros_like(weights_3b)
    zeros_4b = np.zeros_like(weights_4b)
    w3_at_all = np.concatenate([weights_3b, zeros_4b])[order]
    w3x_at_all = np.concatenate([weights_3b * centered, zeros_4b])[order]
    w4_at_all = np.concatenate([zeros_3b, weights_4b])[order]
    cum_w3 = np.cumsum(w3_at_all)
    cum_w3x = np.cumsum(w3x_at_all)
    cdf4 = np.cumsum(w4_at_all) / np.sum(weights_4b)
    total_w3 = np.sum(weights_3b)
    total_w3x = np.sum(weights_3b * centered)

    objectives = np.empty(len(grid))
    for idx, candidate in enumerate(grid):
        cdf3 = (cum_w3 + candidate * cum_w3x) / (
            total_w3 + candidate * total_w3x
        )
        objectives[idx] = np.mean(np.abs(cdf4 - cdf3))

    centered_slope = grid[np.argmin(objectives)]
    slope = centered_slope / std_3b
    intercept = 1 - slope * mean_3b
    return slope, intercept


def load_arrays(hash_):
    tinfo = TrainingInfo.load(hash_)
    cfg = tinfo.hparams["signal_region"]
    signal_filename = tinfo.hparams["dataset"]["signal_filename"]
    stats_train, stats_test = compute_sr_stats(
        cfg["SR_stats_hashes"],
        signal_filename,
        cfg["ensemble_mode"],
        cfg["stats_type"],
    )
    sr_tinfo = TrainingInfo.load(cfg["SR_stats_hashes"][0])
    mother = MotherSamples.load(sr_tinfo.ms_hash)
    events_train = events_from_scdinfo(
        mother.scdinfo[sr_tinfo.ms_idx], FEATURES, signal_filename
    )
    events_test = events_from_scdinfo(
        mother.scdinfo[~sr_tinfo.ms_idx], FEATURES, signal_filename
    )
    sr_cut, _ = get_SR_CR_cut(
        stats_train,
        events_train,
        {"4b_in_SR": cfg["4b_in_SR"], "4b_in_CR": cfg["4b_in_CR"]},
    )
    in_sr = stats_test >= sr_cut
    stats = stats_test[in_sr]
    events = events_test[in_sr]
    scores = tinfo.aux_info["fvt_scores_tst_SR"]
    if len(scores) != len(stats):
        raise ValueError(f"{hash_}: {len(scores)=} does not match {len(stats)=}")
    weights = np.where(
        events.is_4b,
        events.weights,
        scores / (1 - scores) * events.weights,
    )
    is_4b = events.is_4b
    stats_3b = stats[~is_4b]
    stats_4b = stats[is_4b]
    if np.isnan(stats_3b).any() or np.isnan(stats_4b).any():
        raise ValueError(f"{hash_}: NaN values in the SR test statistic")
    n_clipped_3b = int(np.count_nonzero(np.abs(stats_3b) > STATISTIC_CLIP))
    n_clipped_4b = int(np.count_nonzero(np.abs(stats_4b) > STATISTIC_CLIP))
    stats_3b = np.clip(stats_3b, -STATISTIC_CLIP, STATISTIC_CLIP)
    stats_4b = np.clip(stats_4b, -STATISTIC_CLIP, STATISTIC_CLIP)
    return (
        tinfo,
        stats_3b,
        stats_4b,
        weights[~is_4b],
        weights[is_4b],
        n_clipped_3b,
        n_clipped_4b,
    )


def compute_one(hash_):
    start = time.perf_counter()
    (
        tinfo,
        stats_3b,
        stats_4b,
        weights_3b,
        weights_4b,
        n_clipped_3b,
        n_clipped_4b,
    ) = load_arrays(hash_)
    slope, intercept = affine_correction_fast(
        stats_3b, stats_4b, weights_3b, weights_4b
    )
    observed = max_cdf_diff(
        stats_3b,
        stats_4b,
        affine_tilt(stats_3b, weights_3b, slope, intercept),
        weights_4b,
    )

    pooled_stats = np.concatenate([stats_3b, stats_4b])
    pooled_weights = np.concatenate([weights_3b, weights_4b])
    total_weight_3b = np.sum(weights_3b)
    total_weight_4b = np.sum(weights_4b)
    lambda_3b = total_weight_3b / (total_weight_3b + total_weight_4b)
    lambda_4b = total_weight_4b / (total_weight_3b + total_weight_4b)
    rng = np.random.RandomState(0)
    null_values = np.empty(N_REPS)

    for rep in range(N_REPS):
        counts_3b = rng.poisson(lambda_3b, size=len(pooled_stats))
        counts_4b = rng.poisson(lambda_4b, size=len(pooled_stats))
        keep_3b = counts_3b > 0
        keep_4b = counts_4b > 0
        rep_stats_3b = pooled_stats[keep_3b]
        rep_stats_4b = pooled_stats[keep_4b]
        rep_weights_3b = pooled_weights[keep_3b] * counts_3b[keep_3b]
        rep_weights_4b = pooled_weights[keep_4b] * counts_4b[keep_4b]
        rep_slope, rep_intercept = affine_correction_fast(
            rep_stats_3b, rep_stats_4b, rep_weights_3b, rep_weights_4b
        )
        null_values[rep] = max_cdf_diff(
            rep_stats_3b,
            rep_stats_4b,
            affine_tilt(
                rep_stats_3b, rep_weights_3b, rep_slope, rep_intercept
            ),
            rep_weights_4b,
        )

    p_value = np.mean(null_values >= observed)
    return {
        "version": "refit-bootstrap-logitcap10-v1",
        "hash": hash_,
        "seed": tinfo.hparams["dataset"]["seed"],
        "signal_ratio": tinfo.hparams["dataset"]["signal_ratio"],
        "sr_size": tinfo.hparams["signal_region"]["4b_in_SR"],
        "noise_scale": 2.0,
        "n_reps": N_REPS,
        "cdf_mode": CDF_MODE,
        "grid_size": GRID_SIZE,
        "statistic_clip": STATISTIC_CLIP,
        "n_clipped_3b": n_clipped_3b,
        "n_clipped_4b": n_clipped_4b,
        "n_3b": len(stats_3b),
        "n_4b": len(stats_4b),
        "total_weight_3b": total_weight_3b,
        "total_weight_4b": total_weight_4b,
        "lambda_3b": lambda_3b,
        "lambda_4b": lambda_4b,
        "correction_slope": slope,
        "correction_intercept": intercept,
        "observed": observed,
        "null_values": null_values,
        "p_value": p_value,
        "elapsed_seconds": time.perf_counter() - start,
    }


def target_hashes(metadata):
    rows = []
    for hash_, hp in metadata.items():
        if hp.get("experiment_name") != "CR_fvt_training_ensemble_max":
            continue
        if hp.get("dataset", {}).get("signal_ratio") != 0.0:
            continue
        sr_hashes = hp.get("signal_region", {}).get("SR_stats_hashes", [])
        if not sr_hashes:
            continue
        sr_hp = metadata.get(sr_hashes[0], {})
        if sr_hp.get("smearing", {}).get("noise_scale") != 2.0:
            continue
        rows.append(
            (
                hp["signal_region"]["4b_in_SR"],
                hp["dataset"]["seed"],
                hash_,
            )
        )
    rows.sort()
    return [row[2] for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    with open(REPO / "data/metadata/TrainingInfo.pkl", "rb") as handle:
        metadata = pickle.load(handle)
    hashes = target_hashes(metadata)
    if len(hashes) != 400:
        raise RuntimeError(f"Expected 400 eta=2 null hashes, found {len(hashes)}")
    shard = hashes[args.shard_index :: args.n_shards]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"shard={args.shard_index}/{args.n_shards} total_targets={len(hashes)} "
        f"shard_targets={len(shard)}"
    )

    pending = []
    for position, hash_ in enumerate(shard, start=1):
        output_path = OUTPUT_DIR / f"{hash_}.pkl"
        if output_path.exists():
            print(f"skip existing {position}/{len(shard)} {hash_}")
            continue
        pending.append(hash_)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(compute_one, hash_): hash_ for hash_ in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            hash_ = futures[future]
            result = future.result()
            output_path = OUTPUT_DIR / f"{hash_}.pkl"
            temp_path = OUTPUT_DIR / f".{hash_}.{os.getpid()}.tmp"
            with open(temp_path, "wb") as handle:
                pickle.dump(result, handle)
            os.replace(temp_path, output_path)
            print(
                f"done {completed}/{len(pending)} hash={hash_} sr={result['sr_size']} "
                f"seed={result['seed']} p={result['p_value']:.4f} "
                f"seconds={result['elapsed_seconds']:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
