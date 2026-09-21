"""Aggregate the noise-scale sweep into the rows of tab:hypothesis_testing_noise_scale.

Uses the same p-value convention as aggregate_draft_power.py: p = (1+k)/(B+1).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/export/soheuny/SRFinder/soheun")
OUTPUT_DIR = REPO / "data/refit_bootstrap/noise_sweep_logitcap10_v1"
RESULT_DIR = OUTPUT_DIR / "results"
MANIFEST = OUTPUT_DIR / "manifest.pkl"
DETAILED_CSV = OUTPUT_DIR / "noise_sweep_detailed.csv"
SUMMARY_CSV = OUTPUT_DIR / "noise_sweep_summary.csv"

N_REPS = 1000
ALPHA = 0.05
NOISE_SCALES = [0.5, 1.0, 2.0, 3.0]
SIGNAL_RATIOS = [0.0, 0.005, 0.0075, 0.01, 0.02]
SR_SIZES = [0.05, 0.1, 0.15, 0.2]


def p_add_one(k):
    return (1 + k) / (N_REPS + 1)


with open(MANIFEST, "rb") as handle:
    manifest = pickle.load(handle)
expected = {row["hash"] for row in manifest}
paths = list(RESULT_DIR.glob("*.pkl"))
actual = {p.stem for p in paths}
if actual != expected:
    raise RuntimeError(
        f"incomplete: missing={len(expected - actual)} extra={len(actual - expected)}"
    )

rows = []
for path in paths:
    with open(path, "rb") as handle:
        r = pickle.load(handle)
    nv = np.asarray(r["null_values"])
    k_new = int(np.count_nonzero(nv >= r["observed"]))
    if not np.isclose(k_new / N_REPS, r["p_value"]):
        raise RuntimeError(f"{r['hash']}: k mismatch")
    k_old = int(round(r["old_p_value"] * N_REPS))
    rows.append(
        {
            "noise_scale": r["noise_scale"],
            "signal_ratio": r["signal_ratio"],
            "SR_size": r["sr_size"],
            "seed": r["seed"],
            "new_reject": p_add_one(k_new) <= ALPHA,
            "old_reject": p_add_one(k_old) <= ALPHA,
        }
    )

df = pd.DataFrame(rows)
summary = (
    df.groupby(["noise_scale", "signal_ratio", "SR_size"])
    .agg(n=("seed", "size"), new_power=("new_reject", "mean"),
         old_power=("old_reject", "mean"))
    .reset_index()
)
if not (summary["n"] == 100).all():
    raise RuntimeError("not every cell has 100 seeds")

df.to_csv(DETAILED_CSV, index=False)
summary.to_csv(SUMMARY_CSV, index=False)

piv = summary.pivot_table(index=["noise_scale", "SR_size"],
                          columns="signal_ratio", values="new_power")
old = summary.pivot_table(index=["noise_scale", "SR_size"],
                          columns="signal_ratio", values="old_power")

print("=" * 78)
print("REFIT (new) rejection rates -- proposed replacement for the table")
print("=" * 78)
print((piv * 100).round(0).astype(int).to_string())
print()
print("OLD method, same cells (what the table currently shows)")
print((old * 100).round(0).astype(int).to_string())
print()
print("=" * 78)
print("LaTeX rows (refit numbers)")
print("=" * 78)
for ns in NOISE_SCALES:
    for i, sr in enumerate(SR_SIZES):
        vals = " & ".join(
            f"${piv.loc[(ns, sr), eps]:.2f}$" for eps in SIGNAL_RATIOS
        )
        lead = f"\\multirow{{4}}{{*}}{{{ns}}} " if i == 0 else "        "
        print(f"{lead}& ${sr}$ & {vals} \\\\")
    if ns != NOISE_SCALES[-1]:
        print("        \\midrule")
print()
print(f"saved={SUMMARY_CSV}")
