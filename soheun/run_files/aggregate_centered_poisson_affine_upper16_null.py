"""Audit and aggregate the [L,16] affine null campaign."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
DATA_REPO = Path(
    os.environ.get("SRFINDER_DATA_REPO", "/home/export/soheuny/SRFinder/soheun")
)
OUT = DATA_REPO / "data/refit_bootstrap/centered_poisson_affine_upper16_null_v1"
RESULTS = OUT / "results"
VERSION = "centered-poisson1-affine-upper16-null-v1"
BOOTSTRAPS = 1000
ALPHA = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def validate(test: dict, mode: str, hash_: str):
    if test["correction_mode"] != mode or test["bootstrap_replicates"] != BOOTSTRAPS:
        raise RuntimeError(f"{hash_}: invalid {mode} configuration")
    if test["p_value"] != (1 + test["max_exceedances"]) / (BOOTSTRAPS + 1):
        raise RuntimeError(f"{hash_}: invalid {mode} p-value")
    if test["reject"] != (test["p_value"] <= ALPHA):
        raise RuntimeError(f"{hash_}: invalid {mode} rejection")


def main() -> None:
    with (OUT / "manifest.pkl").open("rb") as handle:
        manifest = pickle.load(handle)
    expected = {row["hash"]: row for row in manifest}
    paths = list(RESULTS.glob("*.pkl"))
    if {p.stem for p in paths} != set(expected):
        raise RuntimeError("Incomplete [L,16] campaign")
    hashes = {
        "module_sha256": sha256(CODE_ROOT / "affine_poisson_multiplier_ks.py"),
        "shared_module_sha256": sha256(CODE_ROOT / "poisson_multiplier_ks.py"),
        "base_runner_sha256": sha256(
            CODE_ROOT / "run_files/run_centered_poisson_affine_null.py"
        ),
        "runner_sha256": sha256(
            CODE_ROOT / "run_files/run_centered_poisson_affine_upper16_null.py"
        ),
    }
    rows = []
    for path in paths:
        with path.open("rb") as handle:
            result = pickle.load(handle)
        target = expected[path.stem]
        if result["version"] != VERSION or result["hash"] != path.stem:
            raise RuntimeError(f"{path.stem}: identity/version mismatch")
        for key, value in target.items():
            if result[key] != value:
                raise RuntimeError(f"{path.stem}: manifest mismatch {key}")
        for key, value in hashes.items():
            if result[key] != value:
                raise RuntimeError(f"{path.stem}: source hash mismatch {key}")
        if result["correction_support_upper"] != 16.0:
            raise RuntimeError(f"{path.stem}: wrong upper support")
        validate(result["fixed"], "fixed", path.stem)
        validate(result["composite"], "composite_supremum", path.stem)
        local = result["local_support"]
        fixed = result["fixed"]
        composite = result["composite"]
        rows.append(
            {
                **target,
                "none_reject": local["none_reject"],
                "local_fixed_reject": local["fixed_reject"],
                "local_composite_reject": local["composite_reject"],
                "upper16_fixed_p_value": fixed["p_value"],
                "upper16_fixed_reject": fixed["reject"],
                "upper16_composite_p_value": composite["p_value"],
                "upper16_composite_reject": composite["reject"],
                "fixed_t": fixed["fitted_t"],
                "identity_t": fixed["identity_t"],
                "fixed_lower_boundary": fixed["fitted_t"] <= 1e-12,
                "composite_lower_boundary": composite["maximizing_p_t"] <= 1e-12,
                "fixed_seconds": result["fixed_seconds"],
                "composite_seconds": result["composite_seconds"],
            }
        )
    detailed = pd.DataFrame(rows).sort_values(["noise_scale", "sr_size", "seed"])
    summary = detailed.groupby(["noise_scale", "sr_size"]).agg(
        n=("seed", "size"),
        none_rejections=("none_reject", "sum"),
        none_rejection_rate=("none_reject", "mean"),
        local_fixed_rejections=("local_fixed_reject", "sum"),
        local_fixed_rejection_rate=("local_fixed_reject", "mean"),
        upper16_fixed_rejections=("upper16_fixed_reject", "sum"),
        upper16_fixed_rejection_rate=("upper16_fixed_reject", "mean"),
        local_composite_rejections=("local_composite_reject", "sum"),
        local_composite_rejection_rate=("local_composite_reject", "mean"),
        upper16_composite_rejections=("upper16_composite_reject", "sum"),
        upper16_composite_rejection_rate=("upper16_composite_reject", "mean"),
        mean_upper16_fixed_p_value=("upper16_fixed_p_value", "mean"),
        mean_upper16_composite_p_value=("upper16_composite_p_value", "mean"),
        fixed_lower_boundary_rate=("fixed_lower_boundary", "mean"),
        composite_lower_boundary_rate=("composite_lower_boundary", "mean"),
        mean_fixed_seconds=("fixed_seconds", "mean"),
        mean_composite_seconds=("composite_seconds", "mean"),
    ).reset_index()
    if len(summary) != 8 or not (summary.n == 100).all():
        raise RuntimeError("Expected eight complete cells")
    for prefix in ("none", "local_fixed", "upper16_fixed", "local_composite", "upper16_composite"):
        ci = [interval(int(k), int(n)) for k, n in zip(summary[f"{prefix}_rejections"], summary.n)]
        summary[f"{prefix}_ci_lower95"] = [x[0] for x in ci]
        summary[f"{prefix}_ci_upper95"] = [x[1] for x in ci]
    detailed.to_csv(OUT / "centered_poisson_affine_upper16_null_detailed.csv", index=False)
    summary.to_csv(OUT / "centered_poisson_affine_upper16_null_summary.csv", index=False)
    (OUT / "audit.json").write_text(
        json.dumps(
            {
                "complete": True,
                "completed": len(detailed),
                "expected": len(manifest),
                "cells": len(summary),
                "correction_support": "[SR threshold L, 16]",
                **hashes,
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
