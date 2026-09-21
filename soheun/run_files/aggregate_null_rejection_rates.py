import pickle
from pathlib import Path

import pandas as pd


RESULT_DIR = Path(
    "/home/export/soheuny/SRFinder/soheun/data/refit_bootstrap/"
    "null_eta2_logitcap10_v1"
)
OUTPUT_CSV = RESULT_DIR / "null_rejection_rates_eta2_refit_bootstrap.csv"

rows = []
for path in sorted(RESULT_DIR.glob("*.pkl")):
    with open(path, "rb") as handle:
        result = pickle.load(handle)
    rows.append(
        {
            "hash": result["hash"],
            "seed": result["seed"],
            "SR_size": result["sr_size"],
            "p_value": result["p_value"],
            "reject_at_0.05": result["p_value"] <= 0.05,
            "lambda_3b": result["lambda_3b"],
            "lambda_4b": result["lambda_4b"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
    )

if len(rows) != 400:
    raise RuntimeError(f"Expected 400 completed results, found {len(rows)}")

results = pd.DataFrame(rows).sort_values(["SR_size", "seed"])
counts = results.groupby("SR_size").size()
if not (counts == 100).all():
    raise RuntimeError(f"Expected 100 seeds per SR size, found {counts.to_dict()}")

summary = (
    results.groupby("SR_size")
    .agg(
        rejection_rate=("reject_at_0.05", "mean"),
        n_experiments=("reject_at_0.05", "size"),
        mean_p_value=("p_value", "mean"),
        min_p_value=("p_value", "min"),
        max_p_value=("p_value", "max"),
        mean_lambda_3b=("lambda_3b", "mean"),
        mean_elapsed_seconds=("elapsed_seconds", "mean"),
    )
    .reset_index()
)
summary.to_csv(OUTPUT_CSV, index=False)
print(summary.to_string(index=False))
print(f"saved={OUTPUT_CSV}")
