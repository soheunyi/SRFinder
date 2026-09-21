"""Run fixed/composite affine tests with global support [-16,16]."""

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


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
DATA_REPO = Path(
    os.environ.get("SRFINDER_DATA_REPO", "/home/export/soheuny/SRFinder/soheun")
)
OUT = DATA_REPO / "data/refit_bootstrap/centered_poisson_affine_symmetric16_null_v1"
MANIFEST = OUT / "manifest.pkl"
LOCAL_RESULTS = DATA_REPO / "data/refit_bootstrap/centered_poisson_affine_null_v1/results"
BOOTSTRAPS = 1000
ALPHA = 0.05
SEED = 1729
SUPPORT_LOWER = -16.0
SUPPORT_UPPER = 16.0
NUMERICAL_TOLERANCE = 1e-12
VERSION = "centered-poisson1-affine-symmetric16-null-v1"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_local_result(row: dict) -> dict:
    path = LOCAL_RESULTS / f"{row['hash']}.pkl"
    with path.open("rb") as handle:
        result = pickle.load(handle)
    if result["hash"] != row["hash"] or result["version"] != "centered-poisson1-affine-null-v1":
        raise RuntimeError(f"{row['hash']}: incompatible local-support source")
    return {
        "none_p_value": result["no_correction"]["p_value"],
        "none_reject": result["no_correction"]["reject"],
        "fixed_p_value": result["fixed"]["p_value"],
        "fixed_reject": result["fixed"]["reject"],
        "composite_p_value": result["composite"]["p_value"],
        "composite_reject": result["composite"]["reject"],
        "local_lower": result["lower"],
        "local_upper": result["upper"],
        "source_version": result["version"],
    }


def run_one(row: dict) -> dict:
    from affine_poisson_multiplier_ks import affine_multiplier_ks_test
    from run_files.run_centered_poisson_affine_null import load_arrays_and_cutoff

    started = time.perf_counter()
    scores_3, weights_3, scores_4, weights_4, clipped_3, clipped_4, sr_lower = (
        load_arrays_and_cutoff(row)
    )
    if clipped_3 or clipped_4:
        raise RuntimeError(
            f"{row['hash']}: local source clipped scores; cannot reuse for [-16,16]"
        )
    if scores_3.min() < SUPPORT_LOWER or scores_4.min() < SUPPORT_LOWER:
        raise RuntimeError(f"{row['hash']}: score below symmetric support")
    if scores_3.max() > SUPPORT_UPPER or scores_4.max() > SUPPORT_UPPER:
        raise RuntimeError(f"{row['hash']}: score above symmetric support")
    loaded = time.perf_counter()
    fixed = affine_multiplier_ks_test(
        scores_3,
        weights_3,
        scores_4,
        weights_4,
        L=SUPPORT_LOWER,
        U=SUPPORT_UPPER,
        correction_mode="fixed",
        bootstrap_replicates=BOOTSTRAPS,
        alpha=ALPHA,
        seed=SEED,
        numerical_tolerance=NUMERICAL_TOLERANCE,
    )
    fixed_done = time.perf_counter()
    composite = affine_multiplier_ks_test(
        scores_3,
        weights_3,
        scores_4,
        weights_4,
        L=SUPPORT_LOWER,
        U=SUPPORT_UPPER,
        correction_mode="composite_supremum",
        bootstrap_replicates=BOOTSTRAPS,
        alpha=ALPHA,
        seed=SEED,
        numerical_tolerance=NUMERICAL_TOLERANCE,
    )
    completed = time.perf_counter()
    return {
        **row,
        "version": VERSION,
        "bootstrap_scheme": "independent_centered_poisson1_multiplier",
        "multiplier_stream": "independent_class_seedsequence_spawn_v1",
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "alpha": ALPHA,
        "correction_support_lower": SUPPORT_LOWER,
        "correction_support_upper": SUPPORT_UPPER,
        "sr_threshold": sr_lower,
        "n_clipped_3b": clipped_3,
        "n_clipped_4b": clipped_4,
        "local_support": load_local_result(row),
        "fixed": asdict(fixed),
        "composite": asdict(composite),
        "load_seconds": loaded - started,
        "fixed_seconds": fixed_done - loaded,
        "composite_seconds": completed - fixed_done,
        "module_sha256": file_sha256(CODE_ROOT / "affine_poisson_multiplier_ks.py"),
        "shared_module_sha256": file_sha256(CODE_ROOT / "poisson_multiplier_ks.py"),
        "base_runner_sha256": file_sha256(
            CODE_ROOT / "run_files/run_centered_poisson_affine_null.py"
        ),
        "runner_sha256": file_sha256(Path(__file__)),
    }


def validate_checkpoint(result: dict, row: dict) -> None:
    if result.get("version") != VERSION:
        raise RuntimeError(f"{row['hash']}: incompatible checkpoint version")
    for key, value in row.items():
        if result.get(key) != value:
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
                    "none_p": result["local_support"]["none_p_value"],
                    "local_fixed_p": result["local_support"]["fixed_p_value"],
                    "symmetric_fixed_p": result["fixed"]["p_value"],
                    "local_composite_p": result["local_support"]["composite_p_value"],
                    "symmetric_composite_p": result["composite"]["p_value"],
                    "fixed_t": result["fixed"]["fitted_t"],
                    "identity_t": result["fixed"]["identity_t"],
                    "fixed_seconds": result["fixed_seconds"],
                    "composite_seconds": result["composite_seconds"],
                }
            ),
            flush=True,
        )
        if args.max_new is not None and new_completed >= args.max_new:
            break


if __name__ == "__main__":
    main()
