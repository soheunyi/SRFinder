"""Ten-seed CR validation-closure test with centered Poisson multipliers."""

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
from scipy.stats import beta
import torch
from torch.utils.data import DataLoader


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
DATA_REPO = Path(
    os.environ.get("SRFINDER_DATA_REPO", "/home/export/soheuny/SRFinder/soheun")
)
SOURCE = DATA_REPO / "data/refit_bootstrap/centered_poisson_null_v1"
OUT = DATA_REPO / "data/refit_bootstrap/cr_validation_poisson_null_v1"
SEEDS = set(range(10))
BOOTSTRAPS = 1000
ALPHA = 0.05
RNG_SEED = 1729
STATISTIC_CLIP = 10.0
VERSION = "cr-validation-centered-poisson1-v1"


def configure_data_paths():
    from dataset import MotherSamples
    from training_info import TrainingInfo

    TrainingInfo.SAVE_DIR = DATA_REPO / "data/TrainingInfo"
    TrainingInfo.META_DIR = DATA_REPO / "data/metadata/TrainingInfo.pkl"
    MotherSamples.SAVE_DIR = DATA_REPO / "data/MotherSamples"
    MotherSamples.META_DIR = DATA_REPO / "data/metadata/MotherSamples.pkl"
    return TrainingInfo


def prepare() -> None:
    with (SOURCE / "manifest.pkl").open("rb") as handle:
        rows = [row for row in pickle.load(handle) if row["seed"] in SEEDS]
    rows.sort(key=lambda row: (row["noise_scale"], row["sr_size"], row["seed"]))
    if len(rows) != 80 or len({row["hash"] for row in rows}) != 80:
        raise RuntimeError(f"Expected 80 unique rows, found {len(rows)}")
    counts = pd.DataFrame(rows).groupby(["noise_scale", "sr_size"]).size()
    if len(counts) != 8 or not (counts == 10).all():
        raise RuntimeError("Expected eight cells of ten seeds")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results").mkdir(exist_ok=True)
    path = OUT / "manifest.pkl"
    if path.exists():
        with path.open("rb") as handle:
            if pickle.load(handle) != rows:
                raise RuntimeError("Existing manifest differs")
    else:
        with path.open("xb") as handle:
            pickle.dump(rows, handle)
    (OUT / "manifest_audit.json").write_text(
        json.dumps(
            {
                "campaign": "cr_validation_poisson_null_v1",
                "rows": 80,
                "cells": 8,
                "seeds": sorted(SEEDS),
                "partition": "CR model validation split",
                "warning": "checkpoint selected by validation loss; not an independent test",
                "score": "CR model logit difference",
                "bootstrap": "independent centered Poisson(1) multiplier",
            },
            indent=2,
        )
        + "\n"
    )
    print("prepared 80 CR validation-closure tests")


def load_model(tinfo, device):
    from fvt_classifier import FvTClassifier

    hp = tinfo.hparams
    checkpoint_root = DATA_REPO / "data/checkpoints"
    lightning_path = checkpoint_root / f"{tinfo.hash}_best.ckpt"
    if lightning_path.exists():
        model = FvTClassifier.load_from_checkpoint(
            lightning_path,
            map_location=device,
        )
        model.eval().to(device)
        return model

    model = FvTClassifier(
        num_classes=2,
        dim_input_jet_features=4,
        dim_dijet_features=hp["dim_dijet_features"],
        dim_quadjet_features=hp["dim_quadjet_features"],
        run_name=tinfo.hash,
        device=device,
        depth=hp["depth"],
        repr_norm=hp.get("repr_norm", False),
    )
    path = checkpoint_root / f"{tinfo.hash}_best.pt"
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval().to(device)
    return model


@torch.no_grad()
def predict_logits(model, features, device):
    values = []
    for batch in DataLoader(features, batch_size=2**15, shuffle=False):
        logits = model(batch.to(device))
        values.append((logits[:, 1] - logits[:, 0]).detach().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def run_one(row: dict) -> dict:
    from constants import FEATURES
    from poisson_multiplier_ks import weighted_ks_test

    TrainingInfo = configure_data_paths()
    tinfo = TrainingInfo.load(row["hash"])
    _, val_dataset = tinfo.fetch_train_val_tensor_datasets(
        FEATURES,
        label="fourTag",
        weight="weight",
        label_dtype=torch.long,
    )
    x, labels, physical_weights = val_dataset.tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    model = load_model(tinfo, device)
    logits = predict_logits(model, x, device)
    prediction_seconds = time.perf_counter() - started
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    odds = np.exp(logits)
    labels = labels.numpy()
    physical_weights = physical_weights.numpy().astype(np.float64)
    is_4b = labels == 1
    scores = np.clip(logits, -STATISTIC_CLIP, STATISTIC_CLIP)
    clipped = int(np.count_nonzero(np.abs(logits) > STATISTIC_CLIP))
    weights_3 = physical_weights[~is_4b] * odds[~is_4b]
    weights_4 = physical_weights[is_4b]
    linearized = weighted_ks_test(
        scores[~is_4b],
        weights_3,
        scores[is_4b],
        weights_4,
        bootstrap_replicates=BOOTSTRAPS,
        alpha=ALPHA,
        seed=RNG_SEED,
        form="linearized",
        implementation="cpp",
    )
    direct = weighted_ks_test(
        scores[~is_4b],
        weights_3,
        scores[is_4b],
        weights_4,
        bootstrap_replicates=BOOTSTRAPS,
        alpha=ALPHA,
        seed=RNG_SEED,
        form="direct_normalized",
        implementation="cpp",
    )
    return {
        **row,
        "version": VERSION,
        "partition": "cr_validation",
        "validation_selection_warning": True,
        "bootstrap_replicates": BOOTSTRAPS,
        "alpha": ALPHA,
        "rng_seed": RNG_SEED,
        "score_clip": STATISTIC_CLIP,
        "n_clipped": clipped,
        "n_val": len(labels),
        "n3_val": int(np.count_nonzero(~is_4b)),
        "n4_val": int(np.count_nonzero(is_4b)),
        "W3_val": float(np.sum(physical_weights[~is_4b])),
        "W4_val": float(np.sum(physical_weights[is_4b])),
        "max_abs_logit": float(np.max(np.abs(logits))),
        "max_probability": float(np.max(probabilities)),
        "max_odds": float(np.max(odds)),
        "weight_fraction_p_ge_099": float(
            np.sum(physical_weights[probabilities >= 0.99])
            / np.sum(physical_weights)
        ),
        "linearized": asdict(linearized),
        "direct_normalized": asdict(direct),
        "prediction_seconds": prediction_seconds,
    }


def run(index: int) -> None:
    with (OUT / "manifest.pkl").open("rb") as handle:
        row = pickle.load(handle)[index]
    destination = OUT / "results" / f"{row['hash']}.pkl"
    if destination.exists():
        print(f"skip {row['hash']}")
        return
    result = run_one(row)
    temporary = destination.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(result, handle)
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                "hash": result["hash"],
                "eta": result["noise_scale"],
                "sr": result["sr_size"],
                "seed": result["seed"],
                "p": result["linearized"]["p_value"],
                "direct_p": result["direct_normalized"]["p_value"],
                "max_logit": result["max_abs_logit"],
            }
        ),
        flush=True,
    )


def clopper_pearson(k, n):
    lower = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lower, upper


def aggregate() -> None:
    with (OUT / "manifest.pkl").open("rb") as handle:
        manifest = pickle.load(handle)
    paths = list((OUT / "results").glob("*.pkl"))
    if {p.stem for p in paths} != {row["hash"] for row in manifest}:
        raise RuntimeError("Incomplete CR validation campaign")
    rows = []
    for path in paths:
        with path.open("rb") as handle:
            result = pickle.load(handle)
        test = result["linearized"]
        direct = result["direct_normalized"]
        rows.append(
            {
                "hash": result["hash"],
                "noise_scale": result["noise_scale"],
                "sr_size": result["sr_size"],
                "seed": result["seed"],
                "p_value": test["p_value"],
                "reject": test["reject"],
                "direct_p_value": direct["p_value"],
                "direct_reject": direct["reject"],
                "ks_statistic": test["statistic"],
                "max_abs_logit": result["max_abs_logit"],
                "max_odds": result["max_odds"],
                "weight_fraction_p_ge_099": result["weight_fraction_p_ge_099"],
                "n_clipped": result["n_clipped"],
            }
        )
    detailed = pd.DataFrame(rows).sort_values(["noise_scale", "sr_size", "seed"])
    summary = detailed.groupby(["noise_scale", "sr_size"]).agg(
        n=("seed", "size"),
        rejections=("reject", "sum"),
        rejection_rate=("reject", "mean"),
        direct_rejection_rate=("direct_reject", "mean"),
        mean_p_value=("p_value", "mean"),
        median_p_value=("p_value", "median"),
        max_abs_logit=("max_abs_logit", "max"),
        max_odds=("max_odds", "max"),
        max_weight_fraction_p_ge_099=("weight_fraction_p_ge_099", "max"),
        max_n_clipped=("n_clipped", "max"),
    ).reset_index()
    intervals = [clopper_pearson(int(k), int(n)) for k, n in zip(summary.rejections, summary.n)]
    summary["ci_lower95"] = [x[0] for x in intervals]
    summary["ci_upper95"] = [x[1] for x in intervals]
    detailed.to_csv(OUT / "cr_validation_poisson_null_detailed.csv", index=False)
    summary.to_csv(OUT / "cr_validation_poisson_null_summary.csv", index=False)
    (OUT / "audit.json").write_text(
        json.dumps(
            {
                "complete": True,
                "completed": len(detailed),
                "expected": len(manifest),
                "cells": len(summary),
                "seeds_per_cell": 10,
                "partition": "CR model validation split",
                "validation_selection_warning": True,
            },
            indent=2,
        )
        + "\n"
    )
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
