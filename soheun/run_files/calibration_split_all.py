"""split-0.25 vs baseline across every cell of the noise sweep.

Answers two things at once: whether fitting the correction on a held-out quarter
keeps eta=2 (the operating point) calibrated, and what it costs in power.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/export/soheuny/SRFinder/soheun")
sys.path.insert(0, str(REPO))

from run_files.calibration_variants import bootstrap_p, fit_slope, load_arrays
from run_files.recompute_null_rejection_rates import load_arrays as _la  # noqa

OUTPUT_DIR = REPO / "data/refit_bootstrap/calib_split025_all"
RESULT_DIR = OUTPUT_DIR / "results"
MANIFEST = REPO / "data/refit_bootstrap/noise_sweep_logitcap10_v1/manifest.pkl"
FRACTION = 0.25


def one(hash_):
    _, s3, s4, w3, w4, _, _ = load_arrays(hash_)
    out = {"hash": hash_, "baseline": bootstrap_p(s3, s4, w3, w4, None)}
    rng = np.random.RandomState(12345)
    i3, i4 = rng.permutation(len(s3)), rng.permutation(len(s4))
    n3, n4 = int(round(FRACTION * len(s3))), int(round(FRACTION * len(s4)))
    fit3, tst3 = i3[:n3], i3[n3:]
    fit4, tst4 = i4[:n4], i4[n4:]
    fixed = fit_slope(s3[fit3], s4[fit4], w3[fit3], w4[fit4], cap=None)
    out["split-0.25"] = bootstrap_p(
        s3[tst3], s4[tst4], w3[tst3], w4[tst4], None, fixed=fixed
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    args = ap.parse_args()
    rows = pickle.load(open(MANIFEST, "rb"))
    shard = rows[args.shard_index :: args.n_shards]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(shard, 1):
        path = RESULT_DIR / f"{row['hash']}.pkl"
        if path.exists():
            continue
        res = one(row["hash"])
        res.update(noise_scale=row["noise_scale"], signal_ratio=row["signal_ratio"],
                   sr_size=row["sr_size"], seed=row["seed"])
        tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
        pickle.dump(res, open(tmp, "wb"))
        os.replace(tmp, path)
        if i % 10 == 0:
            print(f"{i}/{len(shard)}", flush=True)


if __name__ == "__main__":
    main()
