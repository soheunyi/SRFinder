"""Held-out calibration audit for base and smoothed density-ratio classifiers."""

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
OUT = DATA_REPO / "data/refit_bootstrap/step2_calibration_audit_v1"
NULL_MANIFEST = DATA_REPO / "data/refit_bootstrap/centered_poisson_null_v1/manifest.pkl"
SEEDS = list(range(10))
ETAS = [np.inf, 2.0, 0.5, 0.1]
N_BINS = 20


def configure_data_paths():
    from dataset import MotherSamples
    from training_info import TrainingInfo

    TrainingInfo.SAVE_DIR = DATA_REPO / "data/TrainingInfo"
    TrainingInfo.META_DIR = DATA_REPO / "data/metadata/TrainingInfo.pkl"
    MotherSamples.SAVE_DIR = DATA_REPO / "data/MotherSamples"
    MotherSamples.META_DIR = DATA_REPO / "data/metadata/MotherSamples.pkl"
    return TrainingInfo, MotherSamples


def cr_hash(seed: int, eta: float, TrainingInfo) -> str:
    if eta in {0.5, 2.0, np.inf}:
        with NULL_MANIFEST.open("rb") as handle:
            rows = pickle.load(handle)
        return next(
            row["hash"]
            for row in rows
            if row["seed"] == seed
            and row["sr_size"] == 0.20
            and (
                (np.isinf(eta) and np.isinf(row["noise_scale"]))
                or row["noise_scale"] == eta
            )
        )
    hashes = TrainingInfo.find(
        {
            "experiment_name": "CR_fvt_training_ensemble_max",
            "dataset": lambda x: (
                x["signal_ratio"] == 0.0
                and x["seed"] == seed
                and x["signal_filename"] == "HH4b_picoAOD.h5"
                and x.get("n_3b") == 1_000_000
            ),
            "signal_region": lambda x: (
                x["4b_in_SR"] == 0.20
                and x["ensemble_mode"] == "max"
                and x["stats_type"] == "smeared"
            ),
        }
    )
    selected = []
    for hash_ in hashes:
        tinfo = TrainingInfo.load(hash_)
        source = TrainingInfo.load(tinfo.hparams["signal_region"]["SR_stats_hashes"][0])
        if source.hparams["smearing"]["noise_scale"] == eta:
            selected.append(hash_)
    if len(selected) != 1:
        raise RuntimeError(f"seed={seed}, eta={eta}: found {len(selected)} CR records")
    return selected[0]


def weighted_calibration(logits, labels, weights):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    order = np.argsort(probabilities, kind="stable")
    groups = np.array_split(order, N_BINS)
    predicted = np.array(
        [np.average(probabilities[g], weights=weights[g]) for g in groups]
    )
    observed = np.array([np.average(labels[g], weights=weights[g]) for g in groups])
    mass = np.array([np.sum(weights[g]) for g in groups])
    errors = np.abs(predicted - observed)
    eps = np.finfo(np.float64).eps
    clipped = np.clip(probabilities, eps, 1 - eps)
    log_loss = -np.sum(
        weights * (labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
    ) / np.sum(weights)
    brier = np.sum(weights * (probabilities - labels) ** 2) / np.sum(weights)
    return {
        "curve_predicted": predicted,
        "curve_observed": observed,
        "curve_mass": mass,
        "ece": float(np.average(errors, weights=mass)),
        "mce": float(np.max(errors)),
        "brier": float(brier),
        "log_loss": float(log_loss),
        "max_abs_logit": float(np.max(np.abs(logits))),
        "max_probability": float(np.max(probabilities)),
        "weight_fraction_p_ge_099": float(
            np.sum(weights[probabilities >= 0.99]) / np.sum(weights)
        ),
        "weight_fraction_p_ge_0999": float(
            np.sum(weights[probabilities >= 0.999]) / np.sum(weights)
        ),
    }


def run_seed(seed: int) -> dict:
    from constants import FEATURES
    from events_data import events_from_scdinfo

    TrainingInfo, MotherSamples = configure_data_paths()
    finite_cr = TrainingInfo.load(cr_hash(seed, 0.5, TrainingInfo))
    step2_hashes = finite_cr.hparams["signal_region"]["SR_stats_hashes"]
    first_step2 = TrainingInfo.load(step2_hashes[0])
    mother = MotherSamples.load(first_step2.ms_hash)
    events_test = events_from_scdinfo(
        mother.scdinfo[~first_step2.ms_idx], FEATURES, "HH4b_picoAOD.h5"
    )
    labels = events_test.is_4b.astype(float)
    weights = events_test.weights.astype(float)
    results = []

    for eta in ETAS:
        record = TrainingInfo.load(cr_hash(seed, eta, TrainingInfo))
        hashes = record.hparams["signal_region"]["SR_stats_hashes"]
        if len(hashes) != 15:
            raise RuntimeError(f"seed={seed}, eta={eta}: expected 15 ensemble members")
        for member, hash_ in enumerate(hashes):
            step2 = TrainingInfo.load(hash_)
            if eta == np.inf:
                base = TrainingInfo.load(step2.hparams["encoder_hash"])
                logits = base.aux_info["base_fvt_logit_tst"]
                model_hash = base.hash
                model_seed = base.hparams["model_seed"]
                model_type = "base"
            else:
                observed_eta = step2.hparams["smearing"]["noise_scale"]
                if observed_eta != eta:
                    raise RuntimeError(
                        f"seed={seed}, member={member}: eta {observed_eta} != {eta}"
                    )
                logits = step2.aux_info["smeared_fvt_logit_tst"]
                model_hash = step2.hash
                model_seed = step2.hparams["model_seed"]
                model_type = "smoothed"
            if len(logits) != len(labels):
                raise RuntimeError(f"{model_hash}: held-out logit length mismatch")
            metrics = weighted_calibration(logits, labels, weights)
            results.append(
                {
                    "seed": seed,
                    "noise_scale": eta,
                    "member": member,
                    "model_seed": model_seed,
                    "model_hash": model_hash,
                    "model_type": model_type,
                    **metrics,
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
                raise RuntimeError("Existing manifest differs")
    else:
        with path.open("xb") as handle:
            pickle.dump(manifest, handle)
    (OUT / "manifest_audit.json").write_text(
        json.dumps(
            {
                "campaign": "step2_calibration_audit_v1",
                "seeds": SEEDS,
                "noise_scales": ["infinity", 2.0, 0.5, 0.1],
                "ensemble_members": 15,
                "partition": "independent mother-sample test split X2",
                "bins": N_BINS,
                "weighting": "physical-event-weighted reliability",
            },
            indent=2,
        )
        + "\n"
    )
    print("prepared ten-seed Step-2 calibration audit")


def run(index: int) -> None:
    seed = SEEDS[index]
    destination = OUT / "results" / f"seed_{seed}.pkl"
    if destination.exists():
        print(f"skip seed={seed}")
        return
    result = run_seed(seed)
    temporary = destination.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(result, handle)
    os.replace(temporary, destination)
    summary = {}
    for eta in ETAS:
        values = [x["ece"] for x in result["results"] if x["noise_scale"] == eta]
        summary[str(eta)] = float(np.median(values))
    print(json.dumps({"seed": seed, "median_ece": summary}), flush=True)


def aggregate() -> None:
    rows = []
    curve_groups: dict[float, list[tuple[np.ndarray, np.ndarray]]] = {
        eta: [] for eta in ETAS
    }
    for seed in SEEDS:
        path = OUT / "results" / f"seed_{seed}.pkl"
        if not path.exists():
            raise RuntimeError(f"missing seed {seed}")
        with path.open("rb") as handle:
            result = pickle.load(handle)
        if result["seed"] != seed or len(result["results"]) != 60:
            raise RuntimeError(f"invalid seed checkpoint {seed}")
        for item in result["results"]:
            curve_groups[item["noise_scale"]].append(
                (item["curve_predicted"], item["curve_observed"])
            )
            rows.append(
                {
                    key: value
                    for key, value in item.items()
                    if not key.startswith("curve_")
                }
            )
    detailed = pd.DataFrame(rows)
    detailed.to_csv(OUT / "step2_calibration_detailed.csv", index=False)
    summary = detailed.groupby(["noise_scale", "model_type"]).agg(
        n_models=("model_hash", "size"),
        median_ece=("ece", "median"),
        q10_ece=("ece", lambda x: x.quantile(0.10)),
        q90_ece=("ece", lambda x: x.quantile(0.90)),
        median_brier=("brier", "median"),
        median_log_loss=("log_loss", "median"),
        q90_mce=("mce", lambda x: x.quantile(0.90)),
        max_abs_logit=("max_abs_logit", "max"),
        max_probability=("max_probability", "max"),
        max_weight_fraction_p_ge_099=("weight_fraction_p_ge_099", "max"),
        max_weight_fraction_p_ge_0999=("weight_fraction_p_ge_0999", "max"),
    ).reset_index()
    summary.to_csv(OUT / "step2_calibration_summary.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharex=True, sharey=True)
    for ax, eta in zip(axes, ETAS):
        curves = curve_groups[eta]
        for predicted, observed in curves:
            ax.plot(predicted, observed, color="0.75", alpha=0.35, linewidth=0.6)
        predicted = np.stack([x[0] for x in curves])
        observed = np.stack([x[1] for x in curves])
        ax.plot(
            np.median(predicted, axis=0),
            np.median(observed, axis=0),
            color="#1f77b4",
            linewidth=2.2,
            label="150-model median",
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        label = r"\infty\ (base)" if np.isinf(eta) else f"{eta:g}"
        ax.set_title(rf"$\eta={label}$")
        ax.set_xlabel("predicted weighted 4b probability")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("observed weighted 4b fraction")
    axes[-1].legend(fontsize=8)
    fig.suptitle(
        "Held-out calibration of base and Step-2 classifiers\n"
        "10 mother seeds x 15 ensemble members; physical-event-weighted bins"
    )
    fig.tight_layout()
    fig.savefig(OUT / "step2_reliability_curves.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "step2_reliability_curves.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [
        detailed[
            np.isinf(detailed.noise_scale)
            if np.isinf(eta)
            else detailed.noise_scale == eta
        ]["ece"].to_numpy()
        for eta in ETAS
    ]
    ax.boxplot(data, labels=["inf base", "2", "0.5", "0.1"], showfliers=True)
    ax.set_ylabel("weighted expected calibration error")
    ax.set_xlabel("noise scale")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "step2_ece_boxplots.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "step2_ece_boxplots.pdf", bbox_inches="tight")
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
