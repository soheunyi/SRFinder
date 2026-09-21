import csv
import pickle
from collections import Counter
from pathlib import Path


REPO = Path("/home/export/soheuny/SRFinder/soheun")
OUTPUT_DIR = REPO / "data/refit_bootstrap/power_pilot_eta2_logitcap10_v1"
TARGET_EXPERIMENTS = {
    "CR_fvt_training_ensemble_max",
    "CR_fvt_training_ensemble_max_HH4b_400",
    "CR_fvt_training_ensemble_max_ZH4b",
}
EXPECTED_TARGETS = 560


with open(REPO / "data/metadata/TrainingInfo.pkl", "rb") as handle:
    metadata = pickle.load(handle)

rows = []
for hash_, hp in metadata.items():
    experiment = hp.get("experiment_name")
    if experiment not in TARGET_EXPERIMENTS:
        continue
    signal_ratio = hp.get("dataset", {}).get("signal_ratio")
    seed = hp.get("dataset", {}).get("seed")
    if signal_ratio is None or signal_ratio <= 0 or seed not in range(10):
        continue
    sr_hashes = hp.get("signal_region", {}).get("SR_stats_hashes", [])
    if not sr_hashes:
        continue
    sr_hp = metadata.get(sr_hashes[0], {})
    if hp["signal_region"]["stats_type"] != "smeared":
        continue
    if sr_hp.get("smearing", {}).get("noise_scale") != 2.0:
        continue
    rows.append(
        {
            "hash": hash_,
            "experiment_name": experiment,
            "signal_ratio": signal_ratio,
            "sr_size": hp["signal_region"]["4b_in_SR"],
            "seed": seed,
            "noise_scale": 2.0,
        }
    )

rows.sort(
    key=lambda row: (
        row["experiment_name"],
        row["signal_ratio"],
        row["sr_size"],
        row["seed"],
        row["hash"],
    )
)
if len(rows) != EXPECTED_TARGETS:
    raise RuntimeError(f"Expected {EXPECTED_TARGETS} pilot targets, found {len(rows)}")
if len({row["hash"] for row in rows}) != len(rows):
    raise RuntimeError("Pilot manifest contains duplicate hashes")

cell_counts = Counter(
    (row["experiment_name"], row["signal_ratio"], row["sr_size"])
    for row in rows
)
if set(cell_counts.values()) != {10}:
    raise RuntimeError(f"Expected 10 seeds in every pilot cell: {cell_counts}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pickle_path = OUTPUT_DIR / "manifest.pkl"
csv_path = OUTPUT_DIR / "manifest.csv"
if pickle_path.exists() or csv_path.exists():
    raise FileExistsError("Pilot manifest already exists; refusing to replace it")
with open(pickle_path, "wb") as handle:
    pickle.dump(rows, handle)
with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"saved {len(rows)} pilot targets across {len(cell_counts)} cells")
print(pickle_path)
print(csv_path)
