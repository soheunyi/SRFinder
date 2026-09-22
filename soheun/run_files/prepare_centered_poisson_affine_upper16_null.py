"""Freeze the [L,16] affine null-calibration manifest."""

from __future__ import annotations

import csv
import json
import os
import pickle
from pathlib import Path


DATA_REPO = Path(
    os.environ.get("SRFINDER_DATA_REPO", "/home/export/soheuny/SRFinder/soheun")
)
SOURCE = DATA_REPO / "data/refit_bootstrap/centered_poisson_affine_null_v1"
OUT = DATA_REPO / "data/refit_bootstrap/centered_poisson_affine_upper16_null_v1"


def main() -> None:
    with (SOURCE / "audit.json").open() as handle:
        audit = json.load(handle)
    if not audit.get("complete") or audit.get("completed") != 800:
        raise RuntimeError("Local-support affine source campaign is incomplete")
    with (SOURCE / "manifest.pkl").open("rb") as handle:
        rows = pickle.load(handle)
    if len(rows) != 800 or len({row["hash"] for row in rows}) != 800:
        raise RuntimeError("Expected 800 unique source rows")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results").mkdir(exist_ok=True)
    path = OUT / "manifest.pkl"
    if path.exists():
        with path.open("rb") as handle:
            if pickle.load(handle) != rows:
                raise RuntimeError("Existing [L,16] manifest differs")
    else:
        with path.open("xb") as handle:
            pickle.dump(rows, handle)
    csv_path = OUT / "manifest.csv"
    if not csv_path.exists():
        with csv_path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUT / "manifest_audit.json").write_text(
        json.dumps(
            {
                "campaign": "centered_poisson_affine_upper16_null_v1",
                "source_campaign": "centered_poisson_affine_null_v1",
                "rows": 800,
                "cells": 8,
                "seeds_per_cell": 100,
                "correction_support": "[SR threshold L, 16]",
                "bootstrap_scheme": "independent_centered_poisson1_multiplier",
                "paired_multiplier_stream": "independent_class_seedsequence_spawn_v1",
            },
            indent=2,
        )
        + "\n"
    )
    print("prepared 800 [L,16] affine tests")


if __name__ == "__main__":
    main()
