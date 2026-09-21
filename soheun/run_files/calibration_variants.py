"""Null-calibration variants at eta=1.0, the one noise scale that over-rejects.

Two families, evaluated on the same loaded arrays so the expensive event loading
is paid once per test:

  split-f : fit the affine correction on a fraction f of the SR sample, then run
            the whole test (observed + bootstrap) on the held-out remainder with
            that correction held FIXED. No refit is needed because the correction
            is independent of the data being tested.
  cap-c   : refit in every replicate as now, but restrict the centred slope grid
            to [-c, c], so the tilt can absorb less of the discrepancy.

baseline reproduces the shipped procedure as a control.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/export/soheuny/SRFinder/soheun")
sys.path.insert(0, str(REPO))

from ks_test import affine_tilt, max_cdf_diff
from run_files.recompute_null_rejection_rates import load_arrays, GRID_SIZE, N_REPS

OUTPUT_DIR = REPO / "data/refit_bootstrap/calib_variants_eta1"
RESULT_DIR = OUTPUT_DIR / "results"
MANIFEST = REPO / "data/refit_bootstrap/noise_sweep_logitcap10_v1/manifest.pkl"

SPLIT_FRACTIONS = [0.25, 0.5]
SLOPE_CAPS = [0.10, 0.05]


def fit_slope(stats_3b, stats_4b, weights_3b, weights_4b, cap=None):
    """affine_correction_fast, with an optional cap on the centred slope."""
    mean_3b = np.sum(weights_3b * stats_3b) / np.sum(weights_3b)
    std_3b = np.sqrt(
        np.sum(weights_3b * (stats_3b - mean_3b) ** 2) / np.sum(weights_3b)
    )
    centered = (stats_3b - mean_3b) / std_3b
    slope_max = -1 / np.min(centered)
    slope_min = -1 / np.max(centered)
    if cap is not None:
        slope_max = min(slope_max, cap)
        slope_min = max(slope_min, -cap)
    grid = GRID_SIZE * np.arange(
        int(slope_min / GRID_SIZE), int(slope_max / GRID_SIZE) + 1
    )
    if len(grid) == 0:
        grid = np.array([0.0])

    order = np.argsort(np.concatenate([stats_3b, stats_4b]))
    w3_at_all = np.concatenate([weights_3b, np.zeros_like(weights_4b)])[order]
    w3x_at_all = np.concatenate(
        [weights_3b * centered, np.zeros_like(weights_4b)]
    )[order]
    w4_at_all = np.concatenate([np.zeros_like(weights_3b), weights_4b])[order]
    cum_w3 = np.cumsum(w3_at_all)
    cum_w3x = np.cumsum(w3x_at_all)
    cdf4 = np.cumsum(w4_at_all) / np.sum(weights_4b)
    total_w3 = np.sum(weights_3b)
    total_w3x = np.sum(weights_3b * centered)

    objectives = np.empty(len(grid))
    for idx, cand in enumerate(grid):
        cdf3 = (cum_w3 + cand * cum_w3x) / (total_w3 + cand * total_w3x)
        objectives[idx] = np.mean(np.abs(cdf4 - cdf3))
    centered_slope = grid[np.argmin(objectives)]
    slope = centered_slope / std_3b
    return slope, 1 - slope * mean_3b


def bootstrap_p(s3, s4, w3, w4, refit_cap, fixed=None, seed=0):
    """p-value. fixed=(slope,intercept) holds the correction; else refit each rep."""
    if fixed is None:
        slope, intercept = fit_slope(s3, s4, w3, w4, cap=refit_cap)
    else:
        slope, intercept = fixed
    observed = max_cdf_diff(s3, s4, affine_tilt(s3, w3, slope, intercept), w4)

    pooled_s = np.concatenate([s3, s4])
    # A fixed correction must be baked into the 3b weights BEFORE pooling, so the
    # corrected weights travel with their events and 3b'/4b' stay exchangeable.
    # The refit path tilts inside each replicate instead.
    w3_pool = w3 if fixed is None else affine_tilt(s3, w3, slope, intercept)
    pooled_w = np.concatenate([w3_pool, w4])
    t3, t4 = np.sum(w3_pool), np.sum(w4)
    l3, l4 = t3 / (t3 + t4), t4 / (t3 + t4)
    rng = np.random.RandomState(seed)
    k = 0
    for _ in range(N_REPS):
        c3 = rng.poisson(l3, size=len(pooled_s))
        c4 = rng.poisson(l4, size=len(pooled_s))
        m3, m4 = c3 > 0, c4 > 0
        r3, r4 = pooled_s[m3], pooled_s[m4]
        rw3, rw4 = pooled_w[m3] * c3[m3], pooled_w[m4] * c4[m4]
        if fixed is None:
            rs, ri = fit_slope(r3, r4, rw3, rw4, cap=refit_cap)
            rw3_corrected = affine_tilt(r3, rw3, rs, ri)
        else:
            rw3_corrected = rw3  # already corrected before pooling
        if max_cdf_diff(r3, r4, rw3_corrected, rw4) >= observed:
            k += 1
    return (1 + k) / (N_REPS + 1)


def one(hash_):
    _, s3, s4, w3, w4, _, _ = load_arrays(hash_)
    out = {"hash": hash_, "baseline": bootstrap_p(s3, s4, w3, w4, None)}

    for cap in SLOPE_CAPS:
        out[f"cap-{cap}"] = bootstrap_p(s3, s4, w3, w4, cap)

    for f in SPLIT_FRACTIONS:
        rng = np.random.RandomState(12345)
        i3 = rng.permutation(len(s3))
        i4 = rng.permutation(len(s4))
        n3, n4 = int(round(f * len(s3))), int(round(f * len(s4)))
        fit3, tst3 = i3[:n3], i3[n3:]
        fit4, tst4 = i4[:n4], i4[n4:]
        fixed = fit_slope(s3[fit3], s4[fit4], w3[fit3], w4[fit4], cap=None)
        out[f"split-{f}"] = bootstrap_p(
            s3[tst3], s4[tst4], w3[tst3], w4[tst4], None, fixed=fixed
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    args = ap.parse_args()

    rows = [
        r for r in pickle.load(open(MANIFEST, "rb"))
        if r["noise_scale"] == 1.0 and r["signal_ratio"] == 0.0
    ]
    shard = rows[args.shard_index :: args.n_shards]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(shard, 1):
        path = RESULT_DIR / f"{row['hash']}.pkl"
        if path.exists():
            continue
        res = one(row["hash"])
        res.update(sr_size=row["sr_size"], seed=row["seed"])
        tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
        pickle.dump(res, open(tmp, "wb"))
        os.replace(tmp, path)
        print(f"{i}/{len(shard)} sr={row['sr_size']} seed={row['seed']} "
              + " ".join(f"{k}={v:.3f}" for k, v in res.items()
                         if isinstance(v, float)), flush=True)


if __name__ == "__main__":
    main()
