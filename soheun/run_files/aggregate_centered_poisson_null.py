"""Audit and aggregate the reduced centered-Poisson null campaign."""

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
OUT = DATA_REPO / "data/refit_bootstrap/centered_poisson_null_v1"
RESULTS = OUT / "results"
VERSION = "centered-poisson1-weighted-ks-v2-cpp"
BOOTSTRAPS = 1000
ALPHA = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(0.025, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(0.975, successes + 1, trials - successes))
    )
    return lower, upper


def validate_test_result(test: dict, *, hash_: str, name: str) -> None:
    if test["bootstrap_replicates"] != BOOTSTRAPS or test["alpha"] != ALPHA:
        raise RuntimeError(f"{hash_}: invalid {name} bootstrap configuration")
    if test["bootstrap_form"] != name or test["implementation"] != "cpp":
        raise RuntimeError(f"{hash_}: invalid {name} implementation metadata")
    values = np.asarray(test["bootstrap_statistics"])
    finite = np.isfinite(values)
    if not np.any(finite):
        raise RuntimeError(f"{hash_}: no finite {name} replicates")
    expected_p = (1 + np.count_nonzero(values[finite] >= test["statistic"])) / (
        1 + np.count_nonzero(finite)
    )
    if test["p_value"] != expected_p:
        raise RuntimeError(f"{hash_}: invalid {name} p-value")
    if test["reject"] != (test["p_value"] <= ALPHA):
        raise RuntimeError(f"{hash_}: invalid {name} rejection indicator")


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

    module_hash = sha256(CODE_ROOT / "poisson_multiplier_ks.py")
    adapter_hash = sha256(CODE_ROOT / "run_files/poisson_multiplier_ks_compiled.py")
    kernel_hash = sha256(CODE_ROOT / "run_files/poisson_multiplier_ks_kernel.cpp")
    runner_hash = sha256(CODE_ROOT / "run_files/run_centered_poisson_null.py")
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
        if (
            result["module_sha256"] != module_hash
            or result["adapter_sha256"] != adapter_hash
            or result["kernel_sha256"] != kernel_hash
            or result["runner_sha256"] != runner_hash
        ):
            raise RuntimeError(f"{hash_}: implementation hash mismatch")
        if (
            result["bootstrap_scheme"]
            != "independent_centered_poisson1_multiplier"
            or result["correction_mode"] != "none"
            or result["bootstrap_replicates"] != BOOTSTRAPS
            or result["alpha"] != ALPHA
        ):
            raise RuntimeError(f"{hash_}: test configuration mismatch")
        validate_test_result(result["linearized"], hash_=hash_, name="linearized")
        validate_test_result(
            result["direct_normalized"],
            hash_=hash_,
            name="direct_normalized",
        )
        linearized = result["linearized"]
        direct = result["direct_normalized"]
        rows.append(
            {
                **target,
                "linearized_p_value": linearized["p_value"],
                "linearized_reject": linearized["reject"],
                "direct_p_value": direct["p_value"],
                "direct_reject": direct["reject"],
                "absolute_p_difference": result["absolute_p_difference"],
                "ks_statistic": linearized["statistic"],
                "n3": linearized["n3"],
                "n4": linearized["n4"],
                "effective_sample_size_3": linearized["effective_sample_size_3"],
                "effective_sample_size_4": linearized["effective_sample_size_4"],
                "max_normalized_weight_3": linearized["max_normalized_weight_3"],
                "max_normalized_weight_4": linearized["max_normalized_weight_4"],
                "n_clipped_3b": result["n_clipped_3b"],
                "n_clipped_4b": result["n_clipped_4b"],
                "load_seconds": result["load_seconds"],
                "bootstrap_seconds": result["bootstrap_seconds"],
            }
        )

    detailed = pd.DataFrame(rows).sort_values(
        ["noise_scale", "sr_size", "seed", "hash"]
    )
    grouped = detailed.groupby(["noise_scale", "sr_size"], sort=True)
    summary = grouped.agg(
        n=("seed", "size"),
        linearized_rejections=("linearized_reject", "sum"),
        linearized_rejection_rate=("linearized_reject", "mean"),
        direct_rejections=("direct_reject", "sum"),
        direct_rejection_rate=("direct_reject", "mean"),
        mean_linearized_p_value=("linearized_p_value", "mean"),
        mean_direct_p_value=("direct_p_value", "mean"),
        mean_absolute_p_difference=("absolute_p_difference", "mean"),
        max_absolute_p_difference=("absolute_p_difference", "max"),
        mean_ks_statistic=("ks_statistic", "mean"),
        mean_effective_sample_size_3=("effective_sample_size_3", "mean"),
        mean_effective_sample_size_4=("effective_sample_size_4", "mean"),
        max_normalized_weight_3=("max_normalized_weight_3", "max"),
        max_normalized_weight_4=("max_normalized_weight_4", "max"),
        max_n_clipped_3b=("n_clipped_3b", "max"),
        max_n_clipped_4b=("n_clipped_4b", "max"),
        mean_load_seconds=("load_seconds", "mean"),
        mean_bootstrap_seconds=("bootstrap_seconds", "mean"),
    ).reset_index()
    if len(summary) != 8 or not (summary["n"] == 100).all():
        raise RuntimeError("Expected eight complete cells of 100 experiments")
    intervals = [
        clopper_pearson(int(k), int(n))
        for k, n in zip(summary["linearized_rejections"], summary["n"])
    ]
    summary["linearized_ci_lower95"] = [x[0] for x in intervals]
    summary["linearized_ci_upper95"] = [x[1] for x in intervals]
    intervals = [
        clopper_pearson(int(k), int(n))
        for k, n in zip(summary["direct_rejections"], summary["n"])
    ]
    summary["direct_ci_lower95"] = [x[0] for x in intervals]
    summary["direct_ci_upper95"] = [x[1] for x in intervals]

    detailed.to_csv(OUT / "centered_poisson_null_detailed.csv", index=False)
    summary.to_csv(OUT / "centered_poisson_null_summary.csv", index=False)
    audit = {
        "complete": True,
        "completed": len(detailed),
        "expected": len(manifest),
        "cells": len(summary),
        "seeds_per_cell": 100,
        "version": VERSION,
        "bootstrap_scheme": "independent_centered_poisson1_multiplier",
        "bootstrap_replicates": BOOTSTRAPS,
        "alpha": ALPHA,
        "module_sha256": module_hash,
        "adapter_sha256": adapter_hash,
        "kernel_sha256": kernel_hash,
        "runner_sha256": runner_hash,
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Validated and aggregated {len(detailed)}/{len(manifest)} results")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
