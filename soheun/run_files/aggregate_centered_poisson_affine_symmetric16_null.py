"""Audit and aggregate the global [-16,16] affine null campaign."""

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
OUT = DATA_REPO / "data/refit_bootstrap/centered_poisson_affine_symmetric16_null_v1"
RESULTS = OUT / "results"
VERSION = "centered-poisson1-affine-symmetric16-null-v1"
BOOTSTRAPS = 1000
ALPHA = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return lower, upper


def validate_affine(result: dict, *, hash_: str, mode: str) -> None:
    if (
        result["correction_mode"] != mode
        or result["bootstrap_replicates"] != BOOTSTRAPS
        or result["alpha"] != ALPHA
        or result["multiplier_stream"] != "independent_class_seedsequence_spawn_v1"
    ):
        raise RuntimeError(f"{hash_}: invalid {mode} configuration")
    expected_p = (1 + result["max_exceedances"]) / (BOOTSTRAPS + 1)
    if result["p_value"] != expected_p:
        raise RuntimeError(f"{hash_}: invalid {mode} p-value")
    if result["reject"] != (result["p_value"] <= ALPHA):
        raise RuntimeError(f"{hash_}: invalid {mode} rejection")
    if mode == "fixed":
        values = np.asarray(result["bootstrap_statistics"])
        count = np.count_nonzero(values + result["numerical_tolerance"] >= result["statistic"])
        if count != result["max_exceedances"]:
            raise RuntimeError(f"{hash_}: invalid fixed exceedance count")
    elif result["bootstrap_statistics"] is not None:
        raise RuntimeError(f"{hash_}: unexpected composite bootstrap array")


def main() -> None:
    with (OUT / "manifest.pkl").open("rb") as handle:
        manifest = pickle.load(handle)
    expected = {row["hash"]: row for row in manifest}
    paths = list(RESULTS.glob("*.pkl"))
    actual = {path.stem for path in paths}
    if actual != set(expected):
        raise RuntimeError(
            f"Incomplete campaign: missing={len(set(expected) - actual)}, "
            f"extra={len(actual - set(expected))}"
        )

    implementation_hashes = {
        "module_sha256": sha256(CODE_ROOT / "affine_poisson_multiplier_ks.py"),
        "shared_module_sha256": sha256(CODE_ROOT / "poisson_multiplier_ks.py"),
        "base_runner_sha256": sha256(
            CODE_ROOT / "run_files/run_centered_poisson_affine_null.py"
        ),
        "runner_sha256": sha256(
            CODE_ROOT / "run_files/run_centered_poisson_affine_symmetric16_null.py"
        ),
    }
    rows = []
    for path in paths:
        with path.open("rb") as handle:
            result = pickle.load(handle)
        hash_ = path.stem
        target = expected[hash_]
        if result["hash"] != hash_ or result["version"] != VERSION:
            raise RuntimeError(f"{hash_}: identity or version mismatch")
        for key, value in target.items():
            if result[key] != value:
                raise RuntimeError(f"{hash_}: manifest mismatch for {key}")
        for key, value in implementation_hashes.items():
            if result[key] != value:
                raise RuntimeError(f"{hash_}: implementation hash mismatch for {key}")
        if (
            result["correction_support_lower"] != -16.0
            or result["correction_support_upper"] != 16.0
            or result["n_clipped_3b"] != 0
            or result["n_clipped_4b"] != 0
        ):
            raise RuntimeError(f"{hash_}: invalid symmetric support metadata")
        validate_affine(result["fixed"], hash_=hash_, mode="fixed")
        validate_affine(result["composite"], hash_=hash_, mode="composite_supremum")
        local = result["local_support"]
        fixed = result["fixed"]
        composite = result["composite"]
        ratio = (
            fixed["correction_value_U"] / fixed["correction_value_L"]
            if fixed["correction_value_L"] > 0
            else np.inf
        )
        rows.append(
            {
                **target,
                "none_p_value": local["none_p_value"],
                "none_reject": local["none_reject"],
                "local_fixed_p_value": local["fixed_p_value"],
                "local_fixed_reject": local["fixed_reject"],
                "local_composite_p_value": local["composite_p_value"],
                "local_composite_reject": local["composite_reject"],
                "symmetric_fixed_p_value": fixed["p_value"],
                "symmetric_fixed_reject": fixed["reject"],
                "symmetric_composite_p_value": composite["p_value"],
                "symmetric_composite_reject": composite["reject"],
                "fixed_t": fixed["fitted_t"],
                "identity_t": fixed["identity_t"],
                "fixed_t_minus_identity": fixed["fitted_t"] - fixed["identity_t"],
                "correction_value_minus16": fixed["correction_value_L"],
                "correction_value_plus16": fixed["correction_value_U"],
                "correction_endpoint_ratio": ratio,
                "fixed_fit_objective": fixed["fit_objective"],
                "composite_maximizing_p_t": composite["maximizing_p_t"],
                "fixed_seconds": result["fixed_seconds"],
                "composite_seconds": result["composite_seconds"],
            }
        )

    detailed = pd.DataFrame(rows).sort_values(["noise_scale", "sr_size", "seed", "hash"])
    grouped = detailed.groupby(["noise_scale", "sr_size"], sort=True)
    summary = grouped.agg(
        n=("seed", "size"),
        none_rejections=("none_reject", "sum"),
        none_rejection_rate=("none_reject", "mean"),
        local_fixed_rejections=("local_fixed_reject", "sum"),
        local_fixed_rejection_rate=("local_fixed_reject", "mean"),
        symmetric_fixed_rejections=("symmetric_fixed_reject", "sum"),
        symmetric_fixed_rejection_rate=("symmetric_fixed_reject", "mean"),
        local_composite_rejections=("local_composite_reject", "sum"),
        local_composite_rejection_rate=("local_composite_reject", "mean"),
        symmetric_composite_rejections=("symmetric_composite_reject", "sum"),
        symmetric_composite_rejection_rate=("symmetric_composite_reject", "mean"),
        mean_symmetric_fixed_p_value=("symmetric_fixed_p_value", "mean"),
        mean_symmetric_composite_p_value=("symmetric_composite_p_value", "mean"),
        mean_fixed_t_minus_identity=("fixed_t_minus_identity", "mean"),
        mean_abs_fixed_t_minus_identity=("fixed_t_minus_identity", lambda x: np.mean(np.abs(x))),
        median_correction_endpoint_ratio=("correction_endpoint_ratio", "median"),
        fixed_lower_boundary_rate=("fixed_t", lambda x: np.mean(x <= 1e-12)),
        fixed_upper_boundary_rate=("fixed_t", lambda x: np.mean(x >= 1 - 1e-12)),
        composite_lower_boundary_rate=("composite_maximizing_p_t", lambda x: np.mean(x <= 1e-12)),
        composite_upper_boundary_rate=("composite_maximizing_p_t", lambda x: np.mean(x >= 1 - 1e-12)),
        mean_fixed_seconds=("fixed_seconds", "mean"),
        mean_composite_seconds=("composite_seconds", "mean"),
    ).reset_index()
    if len(summary) != 8 or not (summary["n"] == 100).all():
        raise RuntimeError("Expected eight complete cells of 100 experiments")

    for prefix in (
        "none",
        "local_fixed",
        "symmetric_fixed",
        "local_composite",
        "symmetric_composite",
    ):
        intervals = [
            clopper_pearson(int(k), int(n))
            for k, n in zip(summary[f"{prefix}_rejections"], summary["n"])
        ]
        summary[f"{prefix}_ci_lower95"] = [x[0] for x in intervals]
        summary[f"{prefix}_ci_upper95"] = [x[1] for x in intervals]

    detailed.to_csv(OUT / "centered_poisson_affine_symmetric16_null_detailed.csv", index=False)
    summary.to_csv(OUT / "centered_poisson_affine_symmetric16_null_summary.csv", index=False)
    audit = {
        "complete": True,
        "completed": len(detailed),
        "expected": len(manifest),
        "cells": len(summary),
        "seeds_per_cell": 100,
        "version": VERSION,
        "correction_support": [-16.0, 16.0],
        "bootstrap_scheme": "independent_centered_poisson1_multiplier",
        "multiplier_stream": "independent_class_seedsequence_spawn_v1",
        "bootstrap_replicates": BOOTSTRAPS,
        "alpha": ALPHA,
        **implementation_hashes,
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Validated and aggregated {len(detailed)}/{len(manifest)} results")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
