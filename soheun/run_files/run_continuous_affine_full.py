"""Run the validated continuous affine-nuisance KS test over the full draft grid."""

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


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "data/refit_bootstrap/continuous_affine_full_v1"
MANIFEST = OUT / "manifest.pkl"
NONNEGATIVE_VERSION = "continuous-affine-ks-poisson-v1"
SIGNED_4B_VERSION = "continuous-affine-ks-poisson-signed4b-v1"
BOOTSTRAPS = 1000
SEED = 1729
ALPHA = 0.05
LOWER_CLIP = -10.0
UPPER_CLIP = 10.0
NUMERICAL_TOLERANCE = 1e-12


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arrays_and_cutoff(row: dict):
    from constants import FEATURES
    from dataset import MotherSamples
    from events_data import events_from_scdinfo
    from run_files.recompute_null_rejection_rates import load_arrays
    from signal_region import compute_sr_stats, get_SR_CR_cut
    from training_info import TrainingInfo

    tinfo, scores_3b, scores_4b, weights_3b, weights_4b, clipped_3b, clipped_4b = (
        load_arrays(row["hash"])
    )
    cfg = tinfo.hparams["signal_region"]
    dataset = tinfo.hparams["dataset"]
    observed_metadata = {
        "experiment_name": tinfo.hparams["experiment_name"],
        "signal_ratio": dataset["signal_ratio"],
        "sr_size": cfg["4b_in_SR"],
        "seed": dataset["seed"],
    }
    for key, observed in observed_metadata.items():
        if observed != row[key]:
            raise RuntimeError(
                f"{row['hash']}: manifest {key}={row[key]} but training info has {observed}"
            )

    signal_filename = dataset["signal_filename"]
    stats_train, _ = compute_sr_stats(
        cfg["SR_stats_hashes"],
        signal_filename,
        cfg["ensemble_mode"],
        cfg["stats_type"],
    )
    source = TrainingInfo.load(cfg["SR_stats_hashes"][0])
    noise_scale = source.hparams["smearing"]["noise_scale"]
    if noise_scale != row["noise_scale"]:
        raise RuntimeError(
            f"{row['hash']}: manifest eta={row['noise_scale']} but source has {noise_scale}"
        )
    mother = MotherSamples.load(source.ms_hash)
    events_train = events_from_scdinfo(
        mother.scdinfo[source.ms_idx], FEATURES, signal_filename
    )
    cutoff, _ = get_SR_CR_cut(
        stats_train,
        events_train,
        {"4b_in_SR": cfg["4b_in_SR"], "4b_in_CR": cfg["4b_in_CR"]},
    )
    lower = float(np.clip(cutoff, LOWER_CLIP, UPPER_CLIP))
    if not lower < UPPER_CLIP:
        raise RuntimeError(f"{row['hash']}: degenerate clipped support [{lower}, 10]")
    return (
        tinfo,
        scores_3b,
        weights_3b,
        scores_4b,
        weights_4b,
        clipped_3b,
        clipped_4b,
        lower,
    )


def run_one(row: dict) -> dict:
    started = time.perf_counter()
    (
        _tinfo,
        scores_3b,
        weights_3b,
        scores_4b,
        weights_4b,
        clipped_3b,
        clipped_4b,
        lower,
    ) = load_arrays_and_cutoff(row)
    has_signed_4b = bool(np.any(weights_4b < 0))
    if has_signed_4b:
        from affine_weighted_ks_signed_compiled import affine_ks_test

        version = SIGNED_4B_VERSION
        implementation_variant = "signed-4b"
        reference_path = REPO / "run_files/affine_weighted_ks_signed_reference.py"
        adapter_path = REPO / "run_files/affine_weighted_ks_signed_compiled.py"
    else:
        from affine_weighted_ks_compiled import affine_ks_test

        version = NONNEGATIVE_VERSION
        implementation_variant = "nonnegative"
        reference_path = REPO / "run_files/affine_weighted_ks_reference.py"
        adapter_path = REPO / "run_files/affine_weighted_ks_compiled.py"
    loaded = time.perf_counter()
    result = affine_ks_test(
        scores_3b,
        weights_3b,
        scores_4b,
        weights_4b,
        L=lower,
        U=UPPER_CLIP,
        B=BOOTSTRAPS,
        alpha=ALPHA,
        seed=SEED,
        numerical_tol=NUMERICAL_TOLERANCE,
    )
    output = asdict(result)
    absolute_4b_total = float(np.sum(np.abs(weights_4b), dtype=np.float64))
    output.update(
        row,
        version=version,
        implementation_variant=implementation_variant,
        has_signed_4b=has_signed_4b,
        negative_4b_count=int(np.count_nonzero(weights_4b < 0)),
        negative_4b_abs_fraction=(
            float(np.sum(np.abs(weights_4b[weights_4b < 0]), dtype=np.float64))
            / absolute_4b_total
            if absolute_4b_total > 0
            else 0.0
        ),
        signed_4b_total=float(np.sum(weights_4b, dtype=np.float64)),
        statistic_clip=UPPER_CLIP,
        lower=lower,
        upper=UPPER_CLIP,
        n_clipped_3b=clipped_3b,
        n_clipped_4b=clipped_4b,
        rng_seed=SEED,
        load_seconds=loaded - started,
        bootstrap_seconds=time.perf_counter() - loaded,
        reference_sha256=file_sha256(reference_path),
        adapter_sha256=file_sha256(adapter_path),
        base_reference_sha256=file_sha256(
            REPO / "run_files/affine_weighted_ks_reference.py"
        ),
        kernel_sha256=file_sha256(REPO / "run_files/affine_envelope_kernel.cpp"),
    )
    return output


def validate_checkpoint(result: dict, row: dict) -> None:
    if (
        result["version"] not in {NONNEGATIVE_VERSION, SIGNED_4B_VERSION}
        or result["bootstrap_replicates"] != BOOTSTRAPS
    ):
        raise RuntimeError(f"{row['hash']}: incompatible existing checkpoint")
    if result["hash"] != row["hash"]:
        raise RuntimeError(f"{row['hash']}: checkpoint hash mismatch")
    for key in (
        "experiment_name",
        "noise_scale",
        "signal_ratio",
        "sr_size",
        "seed",
        "for_noise_table",
        "for_power_figures",
    ):
        if result[key] != row[key]:
            raise RuntimeError(f"{row['hash']}: checkpoint {key} mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new", type=int)
    args = parser.parse_args()
    if not (0 <= args.shard_index < args.n_shards):
        raise ValueError("Require 0 <= shard-index < n-shards")

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
                    key: result[key]
                    for key in (
                        "hash",
                        "experiment_name",
                        "noise_scale",
                        "signal_ratio",
                        "sr_size",
                        "seed",
                        "p_value",
                        "bootstrap_seconds",
                    )
                }
            ),
            flush=True,
        )
        if args.max_new is not None and new_completed >= args.max_new:
            break


if __name__ == "__main__":
    main()
