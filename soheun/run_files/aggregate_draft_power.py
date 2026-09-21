import pickle
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path("/home/export/soheuny/SRFinder/soheun")
OUTPUT_DIR = REPO / "data/refit_bootstrap/draft_eta2_logitcap10_v1"
RESULT_DIR = OUTPUT_DIR / "results"
MANIFEST = OUTPUT_DIR / "manifest.pkl"
DETAILED_CSV = OUTPUT_DIR / "draft_eta2_old_vs_refit_detailed.csv"
SUMMARY_CSV = OUTPUT_DIR / "draft_eta2_old_vs_refit_summary.csv"

# Bootstrap p-value convention.
#   "add_one": p = (1 + k) / (B + 1)   -- Davison & Hinkley; Phipson & Smyth (2010).
#              Cannot return 0, and is the conservative choice.
#   "naive":   p = k / B
# where k = #{T_b >= T_0}. Both are applied to the old and the new method alike,
# so the old-vs-new comparison stays on equal footing.
N_REPS = 1000
P_VALUE_CONVENTION = "add_one"
ALPHA = 0.05


def p_from_count(k, convention=P_VALUE_CONVENTION):
    if convention == "add_one":
        return (1 + k) / (N_REPS + 1)
    if convention == "naive":
        return k / N_REPS
    raise ValueError(f"unknown convention: {convention}")


with open(MANIFEST, "rb") as handle:
    manifest = pickle.load(handle)
expected = {row["hash"] for row in manifest}
paths = list(RESULT_DIR.glob("*.pkl"))
actual = {path.stem for path in paths}
if actual != expected:
    raise RuntimeError(
        f"Draft results incomplete: missing={len(expected - actual)}, "
        f"extra={len(actual - expected)}"
    )

rows = []
for path in paths:
    with open(path, "rb") as handle:
        result = pickle.load(handle)

    # k for the new method comes straight from the stored null distribution.
    null_values = np.asarray(result["null_values"])
    if len(null_values) != N_REPS:
        raise RuntimeError(f"{result['hash']}: expected {N_REPS} replicates")
    k_new = int(np.count_nonzero(null_values >= result["observed"]))
    if not np.isclose(k_new / N_REPS, result["p_value"]):
        raise RuntimeError(
            f"{result['hash']}: recovered k={k_new} disagrees with stored "
            f"p_value={result['p_value']}"
        )

    # The old pipeline used the same B, so its p-value is a multiple of 1/B and
    # k is recoverable exactly.
    k_old = int(round(result["old_p_value"] * N_REPS))
    if not np.isclose(k_old / N_REPS, result["old_p_value"]):
        raise RuntimeError(
            f"{result['hash']}: old p_value={result['old_p_value']} is not a "
            f"multiple of 1/{N_REPS}"
        )

    old_p = p_from_count(k_old)
    new_p = p_from_count(k_new)
    rows.append(
        {
            "experiment_name": result["experiment_name"],
            "hash": result["hash"],
            "seed": result["seed"],
            "signal_ratio": result["signal_ratio"],
            "SR_size": result["sr_size"],
            "old_k": k_old,
            "new_k": k_new,
            "old_p_value": old_p,
            "new_p_value": new_p,
            "old_p_value_naive": k_old / N_REPS,
            "new_p_value_naive": k_new / N_REPS,
            "old_reject": old_p <= ALPHA,
            "new_reject": new_p <= ALPHA,
        }
    )

results = pd.DataFrame(rows).sort_values(
    ["experiment_name", "signal_ratio", "SR_size", "seed", "hash"]
)
cell_sizes = results.groupby(
    ["experiment_name", "signal_ratio", "SR_size"]
).size()
if len(cell_sizes) != 60 or not (cell_sizes == 100).all():
    raise RuntimeError(f"Unexpected draft cell sizes: {cell_sizes.to_dict()}")

summary = (
    results.groupby(["experiment_name", "signal_ratio", "SR_size"])
    .agg(
        n=("seed", "size"),
        old_power=("old_reject", "mean"),
        new_power=("new_reject", "mean"),
        mean_old_p=("old_p_value", "mean"),
        mean_new_p=("new_p_value", "mean"),
    )
    .reset_index()
)
summary["power_change"] = summary["new_power"] - summary["old_power"]
results.to_csv(DETAILED_CSV, index=False)
summary.to_csv(SUMMARY_CSV, index=False)
print(f"p-value convention: {P_VALUE_CONVENTION} (alpha={ALPHA}, B={N_REPS})")
print(summary.to_string(index=False))
print(f"saved={DETAILED_CSV}")
print(f"saved={SUMMARY_CSV}")
