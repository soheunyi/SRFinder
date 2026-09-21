import pickle
from pathlib import Path

import pandas as pd


REPO = Path("/home/export/soheuny/SRFinder/soheun")
OUTPUT_DIR = REPO / "data/refit_bootstrap/power_pilot_eta2_logitcap10_v1"
RESULT_DIR = OUTPUT_DIR / "results"
MANIFEST = OUTPUT_DIR / "manifest.pkl"
OUTPUT_CSV = OUTPUT_DIR / "power_pilot_old_vs_refit.csv"

with open(MANIFEST, "rb") as handle:
    manifest = pickle.load(handle)
expected = {row["hash"] for row in manifest}
paths = list(RESULT_DIR.glob("*.pkl"))
actual = {path.stem for path in paths}
if actual != expected:
    raise RuntimeError(
        f"Pilot incomplete: missing={len(expected - actual)}, extra={len(actual - expected)}"
    )

rows = []
for path in paths:
    with open(path, "rb") as handle:
        result = pickle.load(handle)
    rows.append(
        {
            "experiment_name": result["experiment_name"],
            "signal_ratio": result["signal_ratio"],
            "SR_size": result["sr_size"],
            "seed": result["seed"],
            "old_p_value": result["old_p_value"],
            "new_p_value": result["p_value"],
            "old_reject": result["old_p_value"] <= 0.05,
            "new_reject": result["p_value"] <= 0.05,
        }
    )

results = pd.DataFrame(rows)
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
summary.to_csv(OUTPUT_CSV, index=False)
print(summary.to_string(index=False))
print(f"saved={OUTPUT_CSV}")
