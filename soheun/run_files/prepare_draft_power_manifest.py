import csv
import pickle
from collections import Counter
from pathlib import Path


REPO = Path("/home/export/soheuny/SRFinder/soheun")
OUTPUT_DIR = REPO / "data/refit_bootstrap/draft_eta2_logitcap10_v1"
TARGET_EXPERIMENTS = {
    "CR_fvt_training_ensemble_max",
    "CR_fvt_training_ensemble_max_HH4b_400",
    "CR_fvt_training_ensemble_max_ZH4b",
}
EXPECTED_TARGETS = 6000


with open(REPO / "data/metadata/TrainingInfo.pkl", "rb") as handle:
    metadata = pickle.load(handle)

rows = []
for hash_, hp in metadata.items():
    experiment = hp.get("experiment_name")
    if experiment not in TARGET_EXPERIMENTS:
        continue
    sr_hashes = hp.get("signal_region", {}).get("SR_stats_hashes", [])
    if not sr_hashes or hp["signal_region"]["stats_type"] != "smeared":
        continue
    sr_hp = metadata.get(sr_hashes[0], {})
    if sr_hp.get("smearing", {}).get("noise_scale") != 2.0:
        continue
    rows.append(
        {
            "hash": hash_,
            "experiment_name": experiment,
            "signal_ratio": hp["dataset"]["signal_ratio"],
            "sr_size": hp["signal_region"]["4b_in_SR"],
            "seed": hp["dataset"]["seed"],
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
    raise RuntimeError(f"Expected {EXPECTED_TARGETS} draft targets, found {len(rows)}")
if len({row["hash"] for row in rows}) != len(rows):
    raise RuntimeError("Draft manifest contains duplicate hashes")

cell_counts = Counter(
    (row["experiment_name"], row["signal_ratio"], row["sr_size"])
    for row in rows
)
if len(cell_counts) != 60 or set(cell_counts.values()) != {100}:
    raise RuntimeError(
        f"Expected 60 cells with 100 seeds each; found {len(cell_counts)} cells "
        f"with sizes {sorted(set(cell_counts.values()))}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pickle_path = OUTPUT_DIR / "manifest.pkl"
csv_path = OUTPUT_DIR / "manifest.csv"
if pickle_path.exists() or csv_path.exists():
    raise FileExistsError("Draft manifest already exists; refusing to replace it")
with open(pickle_path, "wb") as handle:
    pickle.dump(rows, handle)
with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"saved {len(rows)} draft targets across {len(cell_counts)} cells")
print(pickle_path)
print(csv_path)
