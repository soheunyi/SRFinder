"""Multi-seed audit of the manuscript CR-to-SR calibration figure.

The manuscript currently shows seed 5.  This diagnostic uses seeds 0,5,...,95,
the same eta values {infinity, 2, 0.1}, and the caption's intended definition:
50 equal-count bins in base gamma with physical-event-weighted means.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
DATA_REPO = Path(
    os.environ.get("SRFINDER_DATA_REPO", "/home/export/soheuny/SRFinder/soheun")
)
OUT = DATA_REPO / "data/refit_bootstrap/calibration_seed_sweep_v1"
CAMPAIGN_NAME = "calibration_seed_sweep_v1"
SEEDS = list(range(0, 100, 5))
ETAS = [np.inf, 2.0, 0.1]
SIGNAL_RATIO = 0.0
SR_SIZE = 0.20
EXPERIMENT = "CR_fvt_training_ensemble_max"
SIGNAL_FILENAME = "HH4b_picoAOD.h5"
N_BINS = 50


def seed_description() -> str:
    if SEEDS == list(range(SEEDS[0], SEEDS[-1] + 1)):
        return f"seeds {SEEDS[0]},...,{SEEDS[-1]}"
    if len(SEEDS) > 2:
        differences = np.diff(SEEDS)
        if np.all(differences == differences[0]):
            return f"seeds {SEEDS[0]},{SEEDS[1]},...,{SEEDS[-1]}"
    return f"{len(SEEDS)} selected seeds"


def configure_data_paths():
    from dataset import MotherSamples
    from training_info import TrainingInfo

    TrainingInfo.SAVE_DIR = DATA_REPO / "data/TrainingInfo"
    TrainingInfo.META_DIR = DATA_REPO / "data/metadata/TrainingInfo.pkl"
    MotherSamples.SAVE_DIR = DATA_REPO / "data/MotherSamples"
    MotherSamples.META_DIR = DATA_REPO / "data/metadata/MotherSamples.pkl"
    return TrainingInfo, MotherSamples


def noise_scale(hash_: str, TrainingInfo) -> float:
    tinfo = TrainingInfo.load(hash_)
    cfg = tinfo.hparams["signal_region"]
    if cfg["stats_type"] == "fvt":
        return np.inf
    source = TrainingInfo.load(cfg["SR_stats_hashes"][0])
    return float(source.hparams["smearing"]["noise_scale"])


def select_hashes(seed: int, TrainingInfo) -> dict[float, str]:
    hashes = TrainingInfo.find(
        {
            "experiment_name": EXPERIMENT,
            "dataset": lambda x: (
                x["signal_ratio"] == SIGNAL_RATIO
                and x["seed"] == seed
                and x["signal_filename"] == SIGNAL_FILENAME
                and x.get("n_3b") == 1_000_000
            ),
            "signal_region": lambda x: (
                x["4b_in_SR"] == SR_SIZE
                and abs(x["4b_in_CR"] - (1.0 - SR_SIZE)) < 1e-12
                and x["ensemble_mode"] == "max"
            ),
        }
    )
    selected: dict[float, str] = {}
    for hash_ in hashes:
        eta = noise_scale(hash_, TrainingInfo)
        if any((np.isinf(eta) and np.isinf(target)) or eta == target for target in ETAS):
            if eta in selected:
                raise RuntimeError(f"seed {seed}: duplicate eta={eta} records")
            selected[eta] = hash_
    if len(selected) != len(ETAS):
        raise RuntimeError(f"seed {seed}: expected eta={ETAS}, found {list(selected)}")
    return selected


def weighted_equal_count_curve(base, predicted, physical_weight):
    base = np.asarray(base, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    physical_weight = np.asarray(physical_weight, dtype=float)
    if not (len(base) == len(predicted) == len(physical_weight)):
        raise RuntimeError("calibration arrays have different lengths")
    if np.any(physical_weight < 0) or not np.sum(physical_weight) > 0:
        raise RuntimeError("expected positive physical weights")
    order = np.argsort(base, kind="stable")
    groups = np.array_split(order, N_BINS)
    x = np.array([np.average(base[g], weights=physical_weight[g]) for g in groups])
    y = np.array([np.average(predicted[g], weights=physical_weight[g]) for g in groups])
    mass = np.array([np.sum(physical_weight[g]) for g in groups])
    return x, y, mass


def run_seed(seed: int) -> dict:
    from constants import FEATURES
    from events_data import events_from_scdinfo
    from signal_region import compute_sr_stats, get_SR_CR_cut

    TrainingInfo, MotherSamples = configure_data_paths()
    hashes = select_hashes(seed, TrainingInfo)
    reference = TrainingInfo.load(hashes[np.inf])
    reference_source = TrainingInfo.load(
        reference.hparams["signal_region"]["SR_stats_hashes"][0]
    )
    mother = MotherSamples.load(reference_source.ms_hash)
    events_train = events_from_scdinfo(
        mother.scdinfo[reference_source.ms_idx], FEATURES, SIGNAL_FILENAME
    )
    events_test = events_from_scdinfo(
        mother.scdinfo[~reference_source.ms_idx], FEATURES, SIGNAL_FILENAME
    )
    first_stats_hash = reference.hparams["signal_region"]["SR_stats_hashes"][0]
    first_stats_info = TrainingInfo.load(first_stats_hash)
    base_info = TrainingInfo.load(first_stats_info.hparams["encoder_hash"])
    base_scores = np.exp(base_info.aux_info["base_fvt_logit_tst"])
    if len(base_scores) != len(events_test):
        raise RuntimeError(f"seed {seed}: base-score/event length mismatch")

    results = []
    for eta in ETAS:
        hash_ = hashes[eta]
        tinfo = TrainingInfo.load(hash_)
        cfg = tinfo.hparams["signal_region"]
        stats_train, stats_test = compute_sr_stats(
            cfg["SR_stats_hashes"],
            SIGNAL_FILENAME,
            cfg["ensemble_mode"],
            cfg["stats_type"],
        )
        sr_cut, _ = get_SR_CR_cut(stats_train, events_train, cfg)
        in_sr = stats_test >= sr_cut
        probabilities = np.asarray(tinfo.aux_info["fvt_scores_tst_SR"], dtype=float)
        if len(probabilities) != np.count_nonzero(in_sr):
            raise RuntimeError(f"seed {seed}, eta={eta}: SR-score length mismatch")
        predicted = probabilities / (1.0 - probabilities)
        x, y, mass = weighted_equal_count_curve(
            base_scores[in_sr],
            predicted,
            events_test[in_sr].weights,
        )
        abs_error = np.abs(y - x)
        results.append(
            {
                "hash": hash_,
                "eta": eta,
                "sr_cut": float(sr_cut),
                "n_sr": int(np.count_nonzero(in_sr)),
                "curve_x": x,
                "curve_y": y,
                "curve_mass": mass,
                "weighted_mae": float(np.average(abs_error, weights=mass)),
                "max_abs_error": float(np.max(abs_error)),
                "signed_weighted_bias": float(np.average(y - x, weights=mass)),
            }
        )
    return {"seed": seed, "results": results}


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results").mkdir(exist_ok=True)
    manifest = [{"seed": seed} for seed in SEEDS]
    path = OUT / "manifest.pkl"
    if path.exists():
        with path.open("rb") as handle:
            if pickle.load(handle) != manifest:
                raise RuntimeError("existing seed-sweep manifest differs")
    else:
        with path.open("xb") as handle:
            pickle.dump(manifest, handle)
    (OUT / "manifest_audit.json").write_text(
        json.dumps(
            {
                "campaign": CAMPAIGN_NAME,
                "seeds": SEEDS,
                "noise_scales": ["infinity", 2.0, 0.1],
                "signal_ratio": SIGNAL_RATIO,
                "sr_size": SR_SIZE,
                "bins": N_BINS,
                "binning": "equal-count by base gamma",
                "averaging": "physical-event-weighted within bin",
                "manuscript_seed": 5,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"prepared {len(manifest)} seeds")


def run(index: int) -> None:
    with (OUT / "manifest.pkl").open("rb") as handle:
        manifest = pickle.load(handle)
    seed = manifest[index]["seed"]
    destination = OUT / "results" / f"seed_{seed}.pkl"
    if destination.exists():
        print(f"skip seed={seed}")
        return
    result = run_seed(seed)
    temporary = destination.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(result, handle)
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                "seed": seed,
                "metrics": {
                    str(row["eta"]): row["weighted_mae"]
                    for row in result["results"]
                },
            }
        ),
        flush=True,
    )


def aggregate() -> None:
    records = []
    curves: dict[float, list[tuple[int, np.ndarray, np.ndarray]]] = {
        eta: [] for eta in ETAS
    }
    for seed in SEEDS:
        path = OUT / "results" / f"seed_{seed}.pkl"
        if not path.exists():
            raise RuntimeError(f"missing seed {seed}")
        with path.open("rb") as handle:
            result = pickle.load(handle)
        if result["seed"] != seed or len(result["results"]) != len(ETAS):
            raise RuntimeError(f"invalid seed checkpoint {seed}")
        for row in result["results"]:
            eta = row["eta"]
            records.append(
                {
                    "seed": seed,
                    "noise_scale": eta,
                    "hash": row["hash"],
                    "sr_cut": row["sr_cut"],
                    "n_sr": row["n_sr"],
                    "weighted_mae": row["weighted_mae"],
                    "max_abs_error": row["max_abs_error"],
                    "signed_weighted_bias": row["signed_weighted_bias"],
                }
            )
            curves[eta].append((seed, row["curve_x"], row["curve_y"]))
    detailed = pd.DataFrame(records)
    detailed.to_csv(OUT / "calibration_seed_sweep_detailed.csv", index=False)

    summary_rows = []
    for eta, group in detailed.groupby("noise_scale", sort=True):
        seed5 = float(group.loc[group.seed == 5, "weighted_mae"].iloc[0])
        values = group["weighted_mae"].to_numpy(float)
        summary_rows.append(
            {
                "noise_scale": eta,
                "n_seeds": len(group),
                "median_weighted_mae": float(np.median(values)),
                "q10_weighted_mae": float(np.quantile(values, 0.10)),
                "q90_weighted_mae": float(np.quantile(values, 0.90)),
                "seed5_weighted_mae": seed5,
                "seed5_percentile": float(np.mean(values <= seed5)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "calibration_seed_sweep_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, eta in zip(axes, ETAS):
        entries = curves[eta]
        for seed, x, y in entries:
            ax.plot(
                x,
                y,
                color="#d62728" if seed == 5 else "0.7",
                alpha=1.0 if seed == 5 else 0.55,
                linewidth=2.0 if seed == 5 else 0.8,
                label="seed 5" if seed == 5 else None,
            )
        stacked_x = np.stack([entry[1] for entry in entries])
        stacked_y = np.stack([entry[2] for entry in entries])
        ax.plot(
            np.median(stacked_x, axis=0),
            np.median(stacked_y, axis=0),
            color="#1f77b4",
            linewidth=2.2,
            label=f"{len(SEEDS)}-seed median",
        )
        lower = min(np.min(stacked_x), np.min(stacked_y))
        upper = max(np.max(stacked_x), np.max(stacked_y))
        ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        eta_label = r"\infty" if np.isinf(eta) else f"{eta:g}"
        ax.set_title(rf"$\eta={eta_label}$, SR")
        ax.set_xlabel(r"$\widehat\gamma$")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"$\widehat\gamma_{\rm CR}$")
    axes[-1].legend(fontsize=8)
    fig.suptitle(
        f"CR-to-SR calibration curves across {seed_description()}\n"
        "50 equal-count bins; physical-event-weighted bin means"
    )
    fig.tight_layout()
    fig.savefig(OUT / "calibration_curves_20_seed_overlay.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "calibration_curves_20_seed_overlay.pdf", bbox_inches="tight")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, eta in zip(axes, ETAS):
        group = detailed[np.isinf(detailed.noise_scale)] if np.isinf(eta) else detailed[detailed.noise_scale == eta]
        values = group["weighted_mae"].to_numpy(float)
        seed5 = float(group.loc[group.seed == 5, "weighted_mae"].iloc[0])
        ax.hist(values, bins=10, color="0.75", edgecolor="0.25")
        ax.axvline(seed5, color="#d62728", linewidth=2, label="seed 5")
        ax.axvline(np.median(values), color="#1f77b4", linewidth=2, label="median")
        eta_label = r"\infty" if np.isinf(eta) else f"{eta:g}"
        ax.set_title(rf"$\eta={eta_label}$")
        ax.set_xlabel("weighted mean absolute calibration error")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("seed count")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "calibration_error_20_seed_histograms.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "calibration_error_20_seed_histograms.pdf", bbox_inches="tight")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--index", type=int)
    group.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    os.chdir(DATA_REPO)
    if args.prepare:
        prepare()
    elif args.aggregate:
        aggregate()
    else:
        run(args.index)


if __name__ == "__main__":
    main()
