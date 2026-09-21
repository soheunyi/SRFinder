"""Report how often the saved pre-cap statistic exceeds its absolute cap."""

from collections import defaultdict
import pickle
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCES = [
    REPO / "data/refit_bootstrap/noise_sweep_logitcap10_v1/results",
    REPO / "data/refit_bootstrap/draft_eta2_logitcap10_v1/results",
]

by_hash = {}
for source in SOURCES:
    for path in source.glob("*.pkl"):
        with path.open("rb") as handle:
            by_hash[path.stem] = pickle.load(handle)
print(f"unique={len(by_hash)}", flush=True)

groups = defaultdict(list)
for result in by_hash.values():
    groups[
        (
            result["experiment_name"],
            result["noise_scale"],
            result["signal_ratio"],
        )
    ].append(result)

for key in sorted(groups):
    rows = groups[key]
    fields = {}
    for label in ("3b", "4b"):
        clipped = sum(row[f"n_clipped_{label}"] for row in rows)
        total = sum(row[f"n_{label}"] for row in rows)
        affected = sum(row[f"n_clipped_{label}"] > 0 for row in rows)
        max_fraction = max(
            row[f"n_clipped_{label}"] / row[f"n_{label}"] for row in rows
        )
        fields[label] = (clipped, total, clipped / total, affected, max_fraction)
    print(f"{key} runs={len(rows)} 3b={fields['3b']} 4b={fields['4b']}", flush=True)
