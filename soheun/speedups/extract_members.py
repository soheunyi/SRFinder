"""Build CR-member-sized training tensors from raw mother samples (no TrainingInfo).

For benchmarking execution-only changes. Each member i uses the null mother
sample with seed i (HH4b, n_3b=1e6, ratio_4b=0.5, signal_ratio=0), then the
same row logic as PR #7:

* X1/X2: artifacts/source_context.VerifiedSourceContext mask
  (base_fvt_train_ratio=0.5, RandomState(dataset seed) shuffle); keep X2.
* a CR-sized slice: 80% of X2 rows (4b_in_CR=0.8). Which rows are CR does not
  matter for execution timing or for exactness checks of execution patches.
* train/validation: artifacts/member_splits.reconstruct_member_splits' Step-3
  branch (val_ratio=0.33, data_seed), training rows aligned to 32.

Model/optimizer/schedule hyperparameters come from the CR_fvt section of the
campaign config. Reads the shared cache only; writes one file per member to --out.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

SOHEUN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOHEUN))
os.chdir(SOHEUN)

from constants import FEATURES  # noqa: E402
from dataset import MotherSamples  # noqa: E402

X1_RATIO, CR_FRACTION, ALIGN = 0.5, 0.8, 32


def member_rows(n: int, seed: int, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(n, dtype=bool)
    mask[:int(n * X1_RATIO)] = True
    np.random.RandomState(seed).shuffle(mask)
    x2 = np.flatnonzero(~mask).astype(np.int64)
    cr = np.sort(np.random.RandomState(seed + 10_000).permutation(x2)[:int(len(x2) * CR_FRACTION)])
    order = np.random.RandomState(seed).permutation(len(cr))
    cut = int((1 - val_ratio) * len(cr))
    train, val = [pd.Series(np.sort(cr[idx])).sample(frac=1, random_state=seed).to_numpy()
                  for idx in (order[:cut], order[cut:])]
    return train[:len(train) // ALIGN * ALIGN], val


def tensors(df: pd.DataFrame, rows: np.ndarray):
    part = df.iloc[rows]
    return (torch.tensor(part[FEATURES].values, dtype=torch.float32),
            torch.tensor(part["fourTag"].values, dtype=torch.long),
            torch.tensor(part["weight"].values, dtype=torch.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=25)
    ap.add_argument("--config", required=True,
                    help="a Step-3 CR config (YAML with CR_fvt and dataset sections)")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(args.config))
    cr = cfg["CR_fvt"]
    dataset_hp = dict(cfg["dataset"])  # n_3b, ratio_4b, signal_ratio, signal_filename
    meta = MotherSamples.load_metadata()
    loaded: dict[pathlib.Path, pd.DataFrame] = {}
    t0 = time.perf_counter()
    for seed in range(args.count):
        target = out / f"member_{seed:03d}.pt"
        if target.exists():
            continue
        want = {**dataset_hp, "seed": seed}
        hashes = sorted(h for h, hp in meta.items() if all(hp.get(k) == v for k, v in want.items()))
        if not hashes:
            raise SystemExit(f"no mother sample for {want}")
        ms = MotherSamples.load(hashes[0])
        for f in ms.scdinfo.files:
            if not any(pathlib.Path(f).resolve() == p.resolve() for p in loaded):
                loaded[pathlib.Path(f)] = pd.read_hdf(f)
        df = ms.scdinfo.fetch_data(loaded).reset_index(drop=True)
        train_rows, val_rows = member_rows(len(df), seed, float(cr["val_ratio"]))
        hparams = {k: cr[k] for k in ("depth", "dim_dijet_features", "dim_quadjet_features", "repr_norm",
                                      "optimizer", "lr_scheduler", "dataloader", "model", "max_epochs",
                                      "val_ratio", "fit_batch_size")}
        hparams.update(model_seed=seed, train_seed=seed, data_seed=seed,
                       benchmark_source={"mother_sample": hashes[0], "dataset": want,
                                         "x1_ratio": X1_RATIO, "cr_fraction": CR_FRACTION})
        record = {"hparams": hparams, "train": tensors(df, train_rows), "val": tensors(df, val_rows)}
        tmp = target.with_suffix(".tmp")
        torch.save(record, tmp)
        os.replace(tmp, target)
        print(f"[extract] {seed}: mother {hashes[0]} rows {len(df)} train {len(train_rows)} "
              f"val {len(val_rows)} ({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
