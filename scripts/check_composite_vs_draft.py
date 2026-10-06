"""Check test 1 (composite) of run_three_tests.py against the draft's stored p.

Uses the draft's range [SR cut, 10], its event order (stable sort by score within
each class) and its multiplier stream (one generator; per replicate all 3b
counts, then all 4b counts). The composite p must equal the draft's p. Includes
signed-4b (ZH4b) configs.
"""
import pickle
import sys

import numpy as np

sys.argv = [sys.argv[0]]
sys.path.insert(0, "/home/export/soheuny/SRFinder/soheun/data/refit_bootstrap/three_tests_obsrange_v1/scripts")
import run_three_tests as rt
from run_files.run_continuous_affine_full import load_arrays_and_cutoff

grid = rt.GRIDS["full"]
rows = pickle.load(open(grid["draft"] / "manifest.pkl", "rb"))
rng = np.random.default_rng(0)
picks = list(rng.choice(len(rows), 8, replace=False)) + [i for i, r in enumerate(rows) if r["experiment_name"].endswith("ZH4b")][:2]
matches = 0
for i in picks:
    row = rows[int(i)]
    _t, s3, w3, s4, w4, _c3, _c4, cut = load_arrays_and_cutoff(row)
    o3, o4 = np.argsort(s3, kind="stable"), np.argsort(s4, kind="stable")
    s3, w3, s4, w4 = s3[o3], w3[o3], s4[o4], w4[o4]
    prepared = rt.prepare(s3, w3, s4, w4, cut, 10.0)
    stream = np.random.default_rng(rt.SEED)
    draw = lambda _r: (stream.poisson(1.0, size=s3.size), stream.poisson(1.0, size=s4.size))
    result = rt.three_tests(prepared, draw)
    draft_p = pickle.load(open(grid["draft"] / "results" / f"{row['hash']}.pkl", "rb"))["p_value"]
    ok = result["composite"]["p_value"] == draft_p
    matches += ok
    print(f"{row['experiment_name'][-12:]:>12} eta={row['noise_scale']} eps={row['signal_ratio']} sr={row['sr_size']} seed={row['seed']} signed={bool(np.any(w4 < 0))}: composite {result['composite']['p_value']:.6f} draft {draft_p:.6f} match={ok}", flush=True)
print(f"composite reproduces draft p in {matches}/{len(picks)} configs", flush=True)
