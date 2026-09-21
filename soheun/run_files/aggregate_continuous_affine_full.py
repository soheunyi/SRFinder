"""Audit and aggregate the full continuous affine manuscript campaign."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/refit_bootstrap/continuous_affine_full_v1"
RESULTS = OUT / "results"
NONNEGATIVE_VERSION = "continuous-affine-ks-poisson-v1"
SIGNED_4B_VERSION = "continuous-affine-ks-poisson-signed4b-v1"
BOOTSTRAPS = 1000
ALPHA = 0.05
BASE = "CR_fvt_training_ensemble_max"
NOISE_SCALES = [0.5, 1.0, 2.0, 3.0]
SIGNAL_RATIOS = [0.0, 0.005, 0.0075, 0.01, 0.02]
SR_SIZES = [0.05, 0.1, 0.15, 0.2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return lower, upper


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

    reference_hash = sha256(REPO / "run_files/affine_weighted_ks_reference.py")
    adapter_hash = sha256(REPO / "run_files/affine_weighted_ks_compiled.py")
    signed_reference_hash = sha256(
        REPO / "run_files/affine_weighted_ks_signed_reference.py"
    )
    signed_adapter_hash = sha256(
        REPO / "run_files/affine_weighted_ks_signed_compiled.py"
    )
    kernel_hash = sha256(REPO / "run_files/affine_envelope_kernel.cpp")
    rows = []
    for path in paths:
        with path.open("rb") as handle:
            result = pickle.load(handle)
        hash_ = path.stem
        target = expected[hash_]
        if result["hash"] != hash_ or result["version"] not in {
            NONNEGATIVE_VERSION,
            SIGNED_4B_VERSION,
        }:
            raise RuntimeError(f"{hash_}: identity or version mismatch")
        for key, value in target.items():
            if result[key] != value:
                raise RuntimeError(f"{hash_}: manifest mismatch for {key}")
        variant = result.get("implementation_variant", "nonnegative")
        if variant == "nonnegative":
            expected_version = NONNEGATIVE_VERSION
            expected_reference = reference_hash
            expected_adapter = adapter_hash
        elif variant == "signed-4b":
            expected_version = SIGNED_4B_VERSION
            expected_reference = signed_reference_hash
            expected_adapter = signed_adapter_hash
            if result.get("base_reference_sha256") != reference_hash:
                raise RuntimeError(f"{hash_}: signed base-reference hash mismatch")
            if not result.get("has_signed_4b") or result.get("negative_4b_count", 0) < 1:
                raise RuntimeError(f"{hash_}: signed implementation without negative weights")
            if not 0 < result.get("negative_4b_abs_fraction", 0) < 1:
                raise RuntimeError(f"{hash_}: invalid negative-weight fraction")
            if not result.get("signed_4b_total", 0) > 0:
                raise RuntimeError(f"{hash_}: nonpositive signed 4b total")
        else:
            raise RuntimeError(f"{hash_}: unknown implementation variant {variant}")
        if result["version"] != expected_version:
            raise RuntimeError(f"{hash_}: version/variant mismatch")
        if (
            result["reference_sha256"] != expected_reference
            or result["adapter_sha256"] != expected_adapter
            or result["kernel_sha256"] != kernel_hash
        ):
            raise RuntimeError(f"{hash_}: implementation hash mismatch")
        if (
            result["bootstrap_replicates"] != BOOTSTRAPS
            or result["statistic_clip"] != 10.0
            or result["alpha"] != ALPHA
            or result["numerical_tolerance"] != 1e-12
        ):
            raise RuntimeError(f"{hash_}: test configuration mismatch")
        expected_p = (1 + result["max_exceedances"]) / (BOOTSTRAPS + 1)
        if result["p_value"] != expected_p:
            raise RuntimeError(f"{hash_}: p-value mismatch")
        if result["reject"] != (result["p_value"] <= ALPHA):
            raise RuntimeError(f"{hash_}: rejection indicator mismatch")
        if not (0 <= result["ks_t"] <= 1 and 0 <= result["maximizing_p_t"] <= 1):
            raise RuntimeError(f"{hash_}: nuisance parameter outside [0,1]")
        if not np.isfinite(result["ks_statistic"]):
            raise RuntimeError(f"{hash_}: nonfinite KS statistic")
        rows.append(
            {
                "hash": hash_,
                "experiment_name": result["experiment_name"],
                "noise_scale": result["noise_scale"],
                "signal_ratio": result["signal_ratio"],
                "SR_size": result["sr_size"],
                "seed": result["seed"],
                "for_noise_table": result["for_noise_table"],
                "for_power_figures": result["for_power_figures"],
                "implementation_variant": variant,
                "negative_4b_count": result.get("negative_4b_count", 0),
                "negative_4b_abs_fraction": result.get(
                    "negative_4b_abs_fraction", 0.0
                ),
                "p_value": result["p_value"],
                "reject": result["reject"],
                "ks_statistic": result["ks_statistic"],
                "ks_t": result["ks_t"],
                "maximizing_p_t": result["maximizing_p_t"],
                "n3": result["n3"],
                "n4": result["n4"],
                "n_clipped_3b": result["n_clipped_3b"],
                "n_clipped_4b": result["n_clipped_4b"],
                "load_seconds": result["load_seconds"],
                "bootstrap_seconds": result["bootstrap_seconds"],
            }
        )

    detailed = pd.DataFrame(rows).sort_values(
        ["experiment_name", "noise_scale", "signal_ratio", "SR_size", "seed", "hash"]
    )
    grouped = detailed.groupby(
        ["experiment_name", "noise_scale", "signal_ratio", "SR_size"],
        sort=True,
    )
    summary = grouped.agg(
        n=("seed", "size"),
        rejections=("reject", "sum"),
        rejection_rate=("reject", "mean"),
        mean_p_value=("p_value", "mean"),
        mean_bootstrap_seconds=("bootstrap_seconds", "mean"),
        mean_load_seconds=("load_seconds", "mean"),
        mean_negative_4b_abs_fraction=("negative_4b_abs_fraction", "mean"),
        max_negative_4b_abs_fraction=("negative_4b_abs_fraction", "max"),
    ).reset_index()
    if len(summary) != 120 or not (summary["n"] == 100).all():
        raise RuntimeError("Expected exactly 120 complete cells of 100 experiments")
    intervals = [
        clopper_pearson(int(k), int(n))
        for k, n in zip(summary["rejections"], summary["n"])
    ]
    summary["ci_lower95"] = [interval[0] for interval in intervals]
    summary["ci_upper95"] = [interval[1] for interval in intervals]

    table = summary[summary["experiment_name"] == BASE].copy()
    if len(table) != 80:
        raise RuntimeError(f"Expected 80 table cells, found {len(table)}")
    power = summary[summary["noise_scale"] == 2.0].copy()
    if len(power) != 60:
        raise RuntimeError(f"Expected 60 power cells, found {len(power)}")

    previous_table = pd.read_csv(
        REPO / "data/refit_bootstrap/noise_sweep_logitcap10_v1/noise_sweep_summary.csv"
    ).rename(columns={"new_power": "previous_refit_rate"})
    table = table.merge(
        previous_table[["noise_scale", "signal_ratio", "SR_size", "previous_refit_rate"]],
        on=["noise_scale", "signal_ratio", "SR_size"],
        how="left",
        validate="one_to_one",
    )
    previous_power = pd.read_csv(
        REPO / "data/refit_bootstrap/draft_eta2_logitcap10_v1/draft_eta2_old_vs_refit_summary.csv"
    ).rename(columns={"new_power": "previous_refit_rate"})
    power = power.merge(
        previous_power[
            ["experiment_name", "signal_ratio", "SR_size", "previous_refit_rate"]
        ],
        on=["experiment_name", "signal_ratio", "SR_size"],
        how="left",
        validate="one_to_one",
    )

    detailed.to_csv(OUT / "continuous_affine_detailed.csv", index=False)
    summary.to_csv(OUT / "continuous_affine_summary.csv", index=False)
    table.to_csv(OUT / "supplementary_noise_scale_summary.csv", index=False)
    power.to_csv(OUT / "power_figure_summary.csv", index=False)

    pivot = table.pivot_table(
        index=["noise_scale", "SR_size"],
        columns="signal_ratio",
        values="rejection_rate",
    )
    lines = []
    for eta in NOISE_SCALES:
        for index, sr_size in enumerate(SR_SIZES):
            values = " & ".join(
                f"${pivot.loc[(eta, sr_size), epsilon]:.2f}$"
                for epsilon in SIGNAL_RATIOS
            )
            lead = f"\\multirow{{4}}{{*}}{{{eta}}} " if index == 0 else "        "
            lines.append(f"{lead}& ${sr_size:g}$ & {values} \\\\")
        if eta != NOISE_SCALES[-1]:
            lines.append("        \\midrule")
    (OUT / "supplementary_noise_scale_table_rows.tex").write_text(
        "\n".join(lines) + "\n"
    )

    audit = {
        "complete": True,
        "completed": len(detailed),
        "expected": len(manifest),
        "cells": len(summary),
        "seeds_per_cell": 100,
        "versions": {
            "nonnegative": NONNEGATIVE_VERSION,
            "signed-4b": SIGNED_4B_VERSION,
        },
        "bootstrap_replicates": BOOTSTRAPS,
        "alpha": ALPHA,
        "statistic_clip": 10.0,
        "reference_sha256": reference_hash,
        "adapter_sha256": adapter_hash,
        "signed_reference_sha256": signed_reference_hash,
        "signed_adapter_sha256": signed_adapter_hash,
        "kernel_sha256": kernel_hash,
        "implementation_counts": {
            str(key): int(value)
            for key, value in detailed["implementation_variant"].value_counts().items()
        },
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("Validated and aggregated 12,000/12,000 results")
    print("Supplementary table rates:")
    print(pivot.to_string())
    print(f"saved={OUT / 'continuous_affine_summary.csv'}")


if __name__ == "__main__":
    main()
