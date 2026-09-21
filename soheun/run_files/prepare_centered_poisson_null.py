"""Freeze the reduced Phase-1 no-correction null-calibration manifest.

The campaign covers nonresonant HH4b, eta in {0.5, 1, 2, infinity}, and
s_SR in {0.05, 0.20}, with 100 seeds per cell.  Metadata and outputs live in
the established SRFinder data checkout, which can be selected with
SRFINDER_DATA_REPO when this script is run from an isolated worktree.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
import os
import pickle
from pathlib import Path


DATA_REPO = Path(
    os.environ.get("SRFINDER_DATA_REPO", "/home/export/soheuny/SRFinder/soheun")
)
OUT = DATA_REPO / "data/refit_bootstrap/centered_poisson_null_v1"
TARGET_EXPERIMENT = "CR_fvt_training_ensemble_max"
SIGNAL_FILENAME = "HH4b_picoAOD.h5"
NOISE_SCALES = {0.5, 1.0, 2.0}
SR_SIZES = {0.05, 0.20}
SEEDS = set(range(100))


def main() -> None:
    metadata_path = DATA_REPO / "data/metadata/TrainingInfo.pkl"
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)

    rows = []
    for hash_, hparams in metadata.items():
        dataset = hparams.get("dataset", {})
        region = hparams.get("signal_region", {})
        if hparams.get("experiment_name") != TARGET_EXPERIMENT:
            continue
        if dataset.get("n_3b") != 1_000_000:
            continue
        if dataset.get("signal_filename") != SIGNAL_FILENAME:
            continue
        if dataset.get("signal_ratio") != 0.0:
            continue
        if dataset.get("seed") not in SEEDS:
            continue
        sr_size = region.get("4b_in_SR")
        if sr_size not in SR_SIZES:
            continue
        if abs(region.get("4b_in_CR", -1.0) - (1.0 - sr_size)) > 1e-12:
            raise RuntimeError(f"{hash_}: CR size is not 1 - SR size")
        if region.get("ensemble_mode") != "max":
            continue
        sr_hashes = region.get("SR_stats_hashes", [])
        if len(sr_hashes) != 15:
            continue

        stats_type = region.get("stats_type")
        if stats_type == "fvt":
            noise_scale = float("inf")
        elif stats_type == "smeared":
            source = metadata.get(sr_hashes[0], {})
            noise_scale = source.get("smearing", {}).get("noise_scale")
            if noise_scale not in NOISE_SCALES:
                continue
        else:
            continue
        rows.append(
            {
                "hash": hash_,
                "experiment_name": TARGET_EXPERIMENT,
                "noise_scale": noise_scale,
                "signal_ratio": 0.0,
                "sr_size": sr_size,
                "seed": dataset["seed"],
                "stats_type": stats_type,
            }
        )

    rows.sort(
        key=lambda row: (
            row["noise_scale"],
            row["sr_size"],
            row["seed"],
            row["hash"],
        )
    )
    cells = Counter((row["noise_scale"], row["sr_size"]) for row in rows)
    expected_scales = NOISE_SCALES | {float("inf")}
    expected_cells = {(eta, sr) for eta in expected_scales for sr in SR_SIZES}
    if set(cells) != expected_cells or set(cells.values()) != {100}:
        raise RuntimeError(
            "Expected eight eta/SR cells with 100 seeds each; found "
            f"{dict(sorted(cells.items()))}"
        )
    if len(rows) != 800 or len({row["hash"] for row in rows}) != 800:
        raise RuntimeError(f"Expected 800 unique rows, found {len(rows)}")
    for cell in expected_cells:
        observed_seeds = {
            row["seed"]
            for row in rows
            if (row["noise_scale"], row["sr_size"]) == cell
        }
        if observed_seeds != SEEDS:
            raise RuntimeError(f"Incomplete seeds for eta/SR cell {cell}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results").mkdir(exist_ok=True)
    pickle_path = OUT / "manifest.pkl"
    csv_path = OUT / "manifest.csv"
    if pickle_path.exists():
        with pickle_path.open("rb") as handle:
            if pickle.load(handle) != rows:
                raise RuntimeError("Existing manifest differs from metadata selection")
    else:
        with pickle_path.open("xb") as handle:
            pickle.dump(rows, handle)
    if not csv_path.exists():
        with csv_path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    audit = {
        "campaign": "centered_poisson_null_v1",
        "rows": len(rows),
        "cells": len(cells),
        "seeds_per_cell": 100,
        "signal_ratio": 0.0,
        "noise_scales": [0.5, 1.0, 2.0, "infinity"],
        "sr_sizes": sorted(SR_SIZES),
        "bootstrap_scheme": "independent_centered_poisson1_multiplier",
        "signal_filename": SIGNAL_FILENAME,
        "n_3b": 1_000_000,
    }
    (OUT / "manifest_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
