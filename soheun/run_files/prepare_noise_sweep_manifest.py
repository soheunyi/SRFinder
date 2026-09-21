"""Manifest for the noise-scale sweep behind tab:hypothesis_testing_noise_scale.

Same CR_fvt_training_ensemble_max grid as the draft scope, but across every noise
scale the table reports rather than eta=2 alone.
"""

import csv
import pickle
from collections import Counter
from pathlib import Path


REPO = Path("/home/export/soheuny/SRFinder/soheun")
OUTPUT_DIR = REPO / "data/refit_bootstrap/noise_sweep_logitcap10_v1"
TARGET_EXPERIMENT = "CR_fvt_training_ensemble_max"
NOISE_SCALES = {0.5, 1.0, 2.0, 3.0}
EXPECTED_TARGETS = 8000
EXPECTED_CELLS = 80

with open(REPO / "data/metadata/TrainingInfo.pkl", "rb") as handle:
    metadata = pickle.load(handle)

rows = []
for hash_, hp in metadata.items():
    if hp.get("experiment_name") != TARGET_EXPERIMENT:
        continue
    sr_hashes = hp.get("signal_region", {}).get("SR_stats_hashes", [])
    if not sr_hashes or hp["signal_region"]["stats_type"] != "smeared":
        continue
    noise_scale = (
        metadata.get(sr_hashes[0], {}).get("smearing", {}).get("noise_scale")
    )
    if noise_scale not in NOISE_SCALES:
        continue
    rows.append(
        {
            "hash": hash_,
            "experiment_name": TARGET_EXPERIMENT,
            "signal_ratio": hp["dataset"]["signal_ratio"],
            "sr_size": hp["signal_region"]["4b_in_SR"],
            "seed": hp["dataset"]["seed"],
            "noise_scale": noise_scale,
        }
    )

rows.sort(
    key=lambda row: (
        row["noise_scale"],
        row["signal_ratio"],
        row["sr_size"],
        row["seed"],
        row["hash"],
    )
)
if len(rows) != EXPECTED_TARGETS:
    raise RuntimeError(f"Expected {EXPECTED_TARGETS} targets, found {len(rows)}")
if len({row["hash"] for row in rows}) != len(rows):
    raise RuntimeError("Manifest contains duplicate hashes")

cell_counts = Counter(
    (row["noise_scale"], row["signal_ratio"], row["sr_size"]) for row in rows
)
if len(cell_counts) != EXPECTED_CELLS or set(cell_counts.values()) != {100}:
    raise RuntimeError(
        f"Expected {EXPECTED_CELLS} cells with 100 seeds each; found "
        f"{len(cell_counts)} cells with sizes {sorted(set(cell_counts.values()))}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pickle_path = OUTPUT_DIR / "manifest.pkl"
csv_path = OUTPUT_DIR / "manifest.csv"
if pickle_path.exists() or csv_path.exists():
    raise FileExistsError("Manifest already exists; refusing to replace it")
with open(pickle_path, "wb") as handle:
    pickle.dump(rows, handle)
with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"saved {len(rows)} targets across {len(cell_counts)} cells")
for ns in sorted(NOISE_SCALES):
    print(f"  eta={ns}: {sum(1 for r in rows if r['noise_scale'] == ns)}")
