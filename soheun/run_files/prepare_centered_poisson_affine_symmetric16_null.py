"""Freeze the global [-16,16] affine null-calibration manifest."""

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
OUT = DATA_REPO / "data/refit_bootstrap/centered_poisson_affine_symmetric16_null_v1"


def main() -> None:
    with (SOURCE / "audit.json").open() as handle:
        source_audit = json.load(handle)
    if not source_audit.get("complete") or source_audit.get("completed") != 800:
        raise RuntimeError("Local-support affine source campaign is not complete")
    with (SOURCE / "manifest.pkl").open("rb") as handle:
        rows = pickle.load(handle)
    if len(rows) != 800 or len({row["hash"] for row in rows}) != 800:
        raise RuntimeError("Expected 800 unique source rows")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results").mkdir(exist_ok=True)
    pickle_path = OUT / "manifest.pkl"
    csv_path = OUT / "manifest.csv"
    if pickle_path.exists():
        with pickle_path.open("rb") as handle:
            if pickle.load(handle) != rows:
                raise RuntimeError("Existing symmetric manifest differs from source")
    else:
        with pickle_path.open("xb") as handle:
            pickle.dump(rows, handle)
    if not csv_path.exists():
        with csv_path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    audit = {
        "campaign": "centered_poisson_affine_symmetric16_null_v1",
        "source_campaign": "centered_poisson_affine_null_v1",
        "rows": len(rows),
        "cells": 8,
        "seeds_per_cell": 100,
        "correction_support": [-16.0, 16.0],
        "correction_modes": ["fixed", "composite_supremum"],
        "bootstrap_scheme": "independent_centered_poisson1_multiplier",
        "paired_multiplier_stream": "independent_class_seedsequence_spawn_v1",
    }
    (OUT / "manifest_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
