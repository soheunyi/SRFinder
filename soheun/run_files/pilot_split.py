"""Pilot: split-fit correction, applied BEFORE pooling, rescaled to preserve the
3b total (affine_tilt returns normalised weights, so the scale must be restored).

Run on the eta=2 cells where the broken version collapsed: eps=0.01 and 0.02,
plus eps=0 for a calibration reference.
"""
import argparse, os, pickle, sys
from pathlib import Path
import numpy as np

REPO = Path("/home/export/soheuny/SRFinder/soheun")
sys.path.insert(0, str(REPO))
from run_files.calibration_variants import fit_slope, load_arrays
from ks_test import affine_tilt, max_cdf_diff

RESULT_DIR = REPO / "data/refit_bootstrap/pilot_split/results"
MANIFEST = REPO / "data/refit_bootstrap/noise_sweep_logitcap10_v1/manifest.pkl"
N_REPS = 1000


def corrected_weights(s, w, slope, intercept):
    """affine_tilt normalises; restore the original total so pooling stays sane."""
    c = affine_tilt(s, w, slope, intercept)
    return c * (w.sum() / c.sum())


def p_refit(s3, s4, w3, w4):
    sl, ic = fit_slope(s3, s4, w3, w4)
    obs = max_cdf_diff(s3, s4, affine_tilt(s3, w3, sl, ic), w4)
    ps, pw = np.concatenate([s3, s4]), np.concatenate([w3, w4])
    t3, t4 = w3.sum(), w4.sum(); l3, l4 = t3/(t3+t4), t4/(t3+t4)
    rng = np.random.RandomState(0); k = 0
    for _ in range(N_REPS):
        c3 = rng.poisson(l3, len(ps)); c4 = rng.poisson(l4, len(ps))
        m3, m4 = c3 > 0, c4 > 0
        r3, r4 = ps[m3], ps[m4]
        rw3, rw4 = pw[m3]*c3[m3], pw[m4]*c4[m4]
        rs, ri = fit_slope(r3, r4, rw3, rw4)
        if max_cdf_diff(r3, r4, affine_tilt(r3, rw3, rs, ri), rw4) >= obs:
            k += 1
    return (1+k)/(N_REPS+1)


def p_split(s3, s4, w3, w4, frac=0.25):
    rng = np.random.RandomState(12345)
    i3, i4 = rng.permutation(len(s3)), rng.permutation(len(s4))
    n3, n4 = int(round(frac*len(s3))), int(round(frac*len(s4)))
    f3, t3i = i3[:n3], i3[n3:]
    f4, t4i = i4[:n4], i4[n4:]
    sl, ic = fit_slope(s3[f3], s4[f4], w3[f3], w4[f4])
    S3, S4 = s3[t3i], s4[t4i]
    W3, W4 = w3[t3i], w4[t4i]
    W3c = corrected_weights(S3, W3, sl, ic)          # rescaled, before pooling
    obs = max_cdf_diff(S3, S4, W3c, W4)
    ps, pw = np.concatenate([S3, S4]), np.concatenate([W3c, W4])
    t3, t4 = W3c.sum(), W4.sum(); l3, l4 = t3/(t3+t4), t4/(t3+t4)
    rng = np.random.RandomState(0); k = 0
    for _ in range(N_REPS):
        c3 = rng.poisson(l3, len(ps)); c4 = rng.poisson(l4, len(ps))
        m3, m4 = c3 > 0, c4 > 0
        if max_cdf_diff(ps[m3], ps[m4], pw[m3]*c3[m3], pw[m4]*c4[m4]) >= obs:
            k += 1
    return (1+k)/(N_REPS+1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    a = ap.parse_args()
    rows = [r for r in pickle.load(open(MANIFEST, "rb"))
            if r["noise_scale"] == 2.0
            and r["signal_ratio"] in (0.0, 0.01, 0.02)
            and r["sr_size"] in (0.05, 0.2)
            and r["seed"] < 15]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for row in rows[a.shard_index::a.n_shards]:
        path = RESULT_DIR / f"{row['hash']}.pkl"
        if path.exists():
            continue
        _, s3, s4, w3, w4, _, _ = load_arrays(row["hash"])
        res = dict(row)
        res["refit"] = p_refit(s3, s4, w3, w4)
        res["split_pre"] = p_split(s3, s4, w3, w4)
        tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
        pickle.dump(res, open(tmp, "wb")); os.replace(tmp, path)
        print(f"eps={row['signal_ratio']} sr={row['sr_size']} seed={row['seed']} "
              f"refit={res['refit']:.3f} split={res['split_pre']:.3f}", flush=True)


if __name__ == "__main__":
    main()
