"""Freeze the deduplicated manifest for all manuscript rejection-rate results."""

from __future__ import annotations

from collections import Counter
import csv
import pickle
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/refit_bootstrap/continuous_affine_full_v1"
TABLE_MANIFEST = (
    REPO / "data/refit_bootstrap/noise_sweep_logitcap10_v1/manifest.pkl"
)
POWER_MANIFEST = (
    REPO / "data/refit_bootstrap/draft_eta2_logitcap10_v1/manifest.pkl"
)


def load(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    table_rows = load(TABLE_MANIFEST)
    power_rows = load(POWER_MANIFEST)
    if len(table_rows) != 8000 or len(power_rows) != 6000:
        raise RuntimeError(
            f"Unexpected source sizes: table={len(table_rows)}, power={len(power_rows)}"
        )

    merged: dict[str, dict] = {}
    for source, rows in (("noise_table", table_rows), ("power_figures", power_rows)):
        for raw in rows:
            row = dict(raw)
            hash_ = row["hash"]
            if hash_ in merged:
                previous = {
                    key: value
                    for key, value in merged[hash_].items()
                    if key not in {"for_noise_table", "for_power_figures"}
                }
                if previous != row:
                    raise RuntimeError(f"Metadata mismatch for shared hash {hash_}")
            else:
                merged[hash_] = row | {
                    "for_noise_table": False,
                    "for_power_figures": False,
                }
            merged[hash_][f"for_{source}"] = True

    rows = sorted(
        merged.values(),
        key=lambda row: (
            row["experiment_name"],
            row["noise_scale"],
            row["signal_ratio"],
            row["sr_size"],
            row["seed"],
            row["hash"],
        ),
    )
    if len(rows) != 12000:
        raise RuntimeError(f"Expected 12,000 unique targets, found {len(rows)}")
    flags = Counter(
        (row["for_noise_table"], row["for_power_figures"]) for row in rows
    )
    expected_flags = Counter({(True, False): 6000, (True, True): 2000, (False, True): 4000})
    if flags != expected_flags:
        raise RuntimeError(f"Unexpected source overlap: {flags}")

    cells = Counter(
        (
            row["experiment_name"],
            row["noise_scale"],
            row["signal_ratio"],
            row["sr_size"],
        )
        for row in rows
    )
    if len(cells) != 120 or set(cells.values()) != {100}:
        raise RuntimeError(
            f"Expected 120 cells with 100 seeds each; got {len(cells)} cells "
            f"with counts {sorted(set(cells.values()))}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results").mkdir(exist_ok=True)
    pickle_path = OUT / "manifest.pkl"
    csv_path = OUT / "manifest.csv"
    if pickle_path.exists():
        with pickle_path.open("rb") as handle:
            if pickle.load(handle) != rows:
                raise RuntimeError("Frozen manifest differs from current source manifests")
    else:
        with pickle_path.open("xb") as handle:
            pickle.dump(rows, handle)
    if not csv_path.exists():
        with csv_path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Frozen {len(rows)} unique hashes across {len(cells)} cells")
    print("table-only=6000 shared=2000 power-only=4000")


if __name__ == "__main__":
    main()
