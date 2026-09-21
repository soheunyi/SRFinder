#!/usr/bin/env python3
"""Is SR signal fraction what drives power, or does psi need to RANK signal high?

Prints S(X_s) at each SR size next to the rejection rate for the same (eta, s_SR).
"""
import os, pickle, sys
from pathlib import Path
REPO = "/home/export/soheuny/SRFinder/soheun"
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, pandas as pd

SR_SIZES = [0.05, 0.10, 0.15, 0.20]
ETAS = [0.5, 1.0, 2.0, 3.0]
grid = np.arange(0, 1.01, 0.01)          # w_4b_ratio_points used when caching

# find the cached sweep for signal_ratio = 0.01
cand = None
for path in sorted(Path("data/figure_cache").glob("signal_concentration.sr_efficiency__*.pkl")):
    with open(path, "rb") as h:
        d = pickle.load(h)
    if not isinstance(d, dict) or "max_sr_stats@2.0" not in d:
        continue
    n_seeds = len(d["max_sr_stats@2.0"])
    # the 0.01 and 0.02 sweeps both have 100 seeds; identify by mean concentration
    cand = cand or []
    cand.append((path, d, n_seeds))

print(f"found {len(cand)} cached sweeps\n")
for path, d, n_seeds in cand:
    print(f"--- {path.name}  ({n_seeds} seeds) ---")
    rows = []
    for eta in ETAS:
        arr = np.asarray(d[f"max_sr_stats@{eta}"])       # (seeds, 101)
        mean = arr.mean(axis=0)
        rows.append([np.interp(s, grid, mean) for s in SR_SIZES])
    print(pd.DataFrame(rows, index=[f"eta={e}" for e in ETAS],
                       columns=[f"S at s_SR={s:g}" for s in SR_SIZES]).round(3).to_string())
    print()

print("=== rejection rate, eps=0.01, from continuous_affine_full_v1 ===")
summ = pd.read_csv("data/refit_bootstrap/continuous_affine_full_v1/continuous_affine_summary.csv")
summ = summ[(summ.experiment_name == "CR_fvt_training_ensemble_max") & (summ.signal_ratio == 0.01)]
print(summ.pivot(index="noise_scale", columns="SR_size", values="rejection_rate").to_string())
