"""No-write signed-weight diagnostics on representative missing ZH rows."""
from dataclasses import asdict
import json
import pickle
from pathlib import Path

import numpy as np

from affine_weighted_ks_signed_compiled import affine_ks_test
from run_continuous_affine_full import load_arrays_and_cutoff


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/refit_bootstrap/continuous_affine_full_v1"
with (OUT / "manifest.pkl").open("rb") as handle:
    manifest = pickle.load(handle)

missing = [
    row
    for row in manifest
    if row["experiment_name"].endswith("ZH4b")
    and not (OUT / "results" / f"{row['hash']}.pkl").exists()
]
selected = []
for ratio in sorted({row["signal_ratio"] for row in missing}):
    selected.append(next(row for row in missing if row["signal_ratio"] == ratio))

for row in selected:
    (
        _tinfo,
        z3,
        w3,
        z4,
        w4,
        _clipped3,
        _clipped4,
        lower,
    ) = load_arrays_and_cutoff(row)
    total = float(w4.sum(dtype=np.float64))
    absolute_total = float(np.abs(w4).sum(dtype=np.float64))
    normalized = w4 / total
    order = np.argsort(z4, kind="stable")
    cumulative = np.cumsum(normalized[order], dtype=np.float64)
    result = affine_ks_test(
        z3,
        w3,
        z4,
        w4,
        L=lower,
        U=10.0,
        B=19,
        alpha=0.05,
        seed=1729,
        numerical_tol=1e-12,
    )
    scaled = affine_ks_test(
        z3,
        7.0 * w3,
        z4,
        11.0 * w4,
        L=lower,
        U=10.0,
        B=19,
        alpha=0.05,
        seed=1729,
        numerical_tol=1e-12,
    )
    assert result.p_value == scaled.p_value
    assert result.max_exceedances == scaled.max_exceedances
    print(
        json.dumps(
            {
                "hash": row["hash"],
                "signal_ratio": row["signal_ratio"],
                "sr_size": row["sr_size"],
                "seed": row["seed"],
                "n4": len(w4),
                "negative_count": int(np.count_nonzero(w4 < 0)),
                "negative_abs_fraction": float(
                    np.abs(w4[w4 < 0]).sum(dtype=np.float64) / absolute_total
                ),
                "signed_to_absolute_total": total / absolute_total,
                "max_abs_normalized_weight": float(np.max(np.abs(normalized))),
                "cumulative_min": float(min(0.0, cumulative.min())),
                "cumulative_max": float(max(1.0, cumulative.max())),
                "p_value_B19": result.p_value,
                "max_exceedances_B19": result.max_exceedances,
            },
            sort_keys=True,
        ),
        flush=True,
    )

print(f"Validated {len(selected)} representative missing ZH rows without writing checkpoints.")

