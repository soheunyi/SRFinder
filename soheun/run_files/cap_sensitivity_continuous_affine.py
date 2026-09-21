"""Paired sensitivity of the continuous affine test to support caps 8, 10, 12."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SOURCE = REPO / "data/refit_bootstrap/continuous_affine_full_v1/manifest.pkl"
OUT = REPO / "data/refit_bootstrap/continuous_affine_cap_sensitivity_v1"
CAPS = (8.0, 10.0, 12.0)
BOOTSTRAPS = 1000
SEED = 1729


def selected(row: dict) -> bool:
    if row["sr_size"] != 0.2 or row["seed"] not in {0, 1}:
        return False
    experiment = row["experiment_name"]
    epsilon = row["signal_ratio"]
    eta = row["noise_scale"]
    if experiment == "CR_fvt_training_ensemble_max":
        return (epsilon == 0.0 and eta in {0.5, 1.0, 2.0, 3.0}) or (
            eta in {1.0, 2.0} and epsilon in {0.0075, 0.01}
        )
    if experiment == "CR_fvt_training_ensemble_max_HH4b_400":
        return eta == 2.0 and epsilon in {0.005, 0.0075}
    if experiment == "CR_fvt_training_ensemble_max_ZH4b":
        return eta == 2.0 and epsilon in {0.02, 0.03, 0.05}
    return False


def prepare() -> None:
    with SOURCE.open("rb") as handle:
        rows = [row for row in pickle.load(handle) if selected(row)]
    rows.sort(
        key=lambda row: (
            row["experiment_name"],
            row["noise_scale"],
            row["signal_ratio"],
            row["seed"],
        )
    )
    if len(rows) != 26 or len({row["hash"] for row in rows}) != 26:
        raise RuntimeError(f"Expected 26 paired datasets, found {len(rows)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results").mkdir(exist_ok=True)
    path = OUT / "manifest.pkl"
    if path.exists():
        with path.open("rb") as handle:
            if pickle.load(handle) != rows:
                raise RuntimeError("Existing sensitivity manifest differs")
    else:
        with path.open("xb") as handle:
            pickle.dump(rows, handle)
    print(f"prepared {len(rows)} paired datasets x {len(CAPS)} caps")


def load_raw(row: dict):
    from constants import FEATURES
    from dataset import MotherSamples
    from events_data import events_from_scdinfo
    from signal_region import compute_sr_stats, get_SR_CR_cut
    from training_info import TrainingInfo

    tinfo = TrainingInfo.load(row["hash"])
    cfg = tinfo.hparams["signal_region"]
    signal_filename = tinfo.hparams["dataset"]["signal_filename"]
    stats_train, stats_test = compute_sr_stats(
        cfg["SR_stats_hashes"], signal_filename, cfg["ensemble_mode"], cfg["stats_type"]
    )
    source = TrainingInfo.load(cfg["SR_stats_hashes"][0])
    mother = MotherSamples.load(source.ms_hash)
    events_train = events_from_scdinfo(
        mother.scdinfo[source.ms_idx], FEATURES, signal_filename
    )
    events_test = events_from_scdinfo(
        mother.scdinfo[~source.ms_idx], FEATURES, signal_filename
    )
    cutoff, _ = get_SR_CR_cut(
        stats_train,
        events_train,
        {"4b_in_SR": cfg["4b_in_SR"], "4b_in_CR": cfg["4b_in_CR"]},
    )
    in_sr = stats_test >= cutoff
    raw_scores = stats_test[in_sr]
    events = events_test[in_sr]
    classifier_scores = tinfo.aux_info["fvt_scores_tst_SR"]
    if len(classifier_scores) != len(raw_scores):
        raise RuntimeError(f"{row['hash']}: classifier-score length mismatch")
    weights = np.where(
        events.is_4b,
        events.weights,
        classifier_scores / (1 - classifier_scores) * events.weights,
    )
    if np.isnan(raw_scores).any():
        raise RuntimeError(f"{row['hash']}: NaN raw statistic")
    return raw_scores[~events.is_4b], weights[~events.is_4b], raw_scores[events.is_4b], weights[events.is_4b], float(cutoff)


def run(index: int) -> None:
    from affine_weighted_ks_compiled import affine_ks_test

    with (OUT / "manifest.pkl").open("rb") as handle:
        rows = pickle.load(handle)
    row = rows[index]
    destination = OUT / "results" / f"{row['hash']}.pkl"
    if destination.exists():
        print(f"skip {row['hash']}")
        return
    raw_3b, weights_3b, raw_4b, weights_4b, cutoff = load_raw(row)
    results = []
    for cap in CAPS:
        lower = float(np.clip(cutoff, -cap, cap))
        if not lower < cap:
            raise RuntimeError(f"{row['hash']}: degenerate support for cap {cap}")
        started = time.perf_counter()
        result = affine_ks_test(
            np.clip(raw_3b, -cap, cap),
            weights_3b,
            np.clip(raw_4b, -cap, cap),
            weights_4b,
            L=lower,
            U=cap,
            B=BOOTSTRAPS,
            alpha=0.05,
            seed=SEED,
        )
        values = asdict(result)
        values.update(
            cap=cap,
            lower=lower,
            n_clipped_3b=int(np.count_nonzero(np.abs(raw_3b) > cap)),
            n_clipped_4b=int(np.count_nonzero(np.abs(raw_4b) > cap)),
            seconds=time.perf_counter() - started,
        )
        results.append(values)
    output = {"row": row, "results": results, "bootstrap_replicates": BOOTSTRAPS}
    temporary = destination.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(output, handle)
    os.replace(temporary, destination)
    print(json.dumps({"hash": row["hash"], "p": {x["cap"]: x["p_value"] for x in results}}))


def aggregate() -> None:
    with (OUT / "manifest.pkl").open("rb") as handle:
        manifest = pickle.load(handle)
    rows = []
    for target in manifest:
        path = OUT / "results" / f"{target['hash']}.pkl"
        if not path.exists():
            raise RuntimeError(f"Missing {path.name}")
        with path.open("rb") as handle:
            output = pickle.load(handle)
        if output["row"] != target or output["bootstrap_replicates"] != BOOTSTRAPS:
            raise RuntimeError(f"Invalid checkpoint {target['hash']}")
        for result in output["results"]:
            rows.append(target | result)
    detailed = pd.DataFrame(rows)
    detailed.to_csv(OUT / "cap_sensitivity_detailed.csv", index=False)
    wide_p = detailed.pivot(index="hash", columns="cap", values="p_value")
    wide_r = detailed.pivot(index="hash", columns="cap", values="reject")
    summary = {
        "datasets": len(wide_p),
        "maximum_absolute_p_difference_8_vs_10": float((wide_p[8.0] - wide_p[10.0]).abs().max()),
        "maximum_absolute_p_difference_10_vs_12": float((wide_p[10.0] - wide_p[12.0]).abs().max()),
        "decision_flips_8_vs_10": int((wide_r[8.0] != wide_r[10.0]).sum()),
        "decision_flips_10_vs_12": int((wide_r[10.0] != wide_r[12.0]).sum()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(detailed[["experiment_name", "noise_scale", "signal_ratio", "seed", "cap", "p_value", "reject", "ks_t", "maximizing_p_t", "n_clipped_3b", "n_clipped_4b"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--index", type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if sum((args.prepare, args.index is not None, args.aggregate)) != 1:
        raise ValueError("Choose exactly one of --prepare, --index, --aggregate")
    if args.prepare:
        prepare()
    elif args.aggregate:
        aggregate()
    else:
        run(args.index)


if __name__ == "__main__":
    main()
