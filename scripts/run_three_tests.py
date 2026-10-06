"""Three affine-correction KS tests on every config the paper uses.

All three use the same family and the same multipliers. Scores are clipped to
[SR cut, 10] by the draft loaders (the upper clip is a sanity guard only). The
affine correction of the 3b weights must be nonnegative on the observed 3b range
[a, b] = [min 3b score, max 3b score]; the corrected normalized 3b CDF is
F_minus + t (F_plus - F_minus), t in [0, 1]. 4b weights may be signed with a
positive total, as in the draft's signed-4b variant.

1. composite: p(t) from the fixed-t centered Poisson(1) bootstrap; p = max over
   t in [0, 1] (the draft method, Berger-Boos), computed exactly with the
   draft's envelope code.
2. fixed:     t_hat = argmin_t sup|D(t)|; statistic sup|D(t_hat)|; bootstrap
   statistic sup|G*(t_hat)| with no refit.
3. refit:     same observed statistic; each replicate recentres at the fitted
   null, G*(t) + (t - t_hat) d_hat, and refits t (L-infinity literal).

Multipliers: SeedSequence(1729).spawn(2) class streams, as in
centered_poisson_obsrange_linf_refit_full_v1, so test 3 must reproduce that run's
p-values exactly; each result stores both for the check.

--grid full     the 12,000 configs of continuous_affine_full_v1
--grid eta_inf  the 2,000 configs of continuous_affine_eta_inf_v1 (internal)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import time

import numpy as np


REPO = Path("/home/export/soheuny/SRFinder/soheun")
GRIDS = {
    "full": {
        "draft": REPO / "data/refit_bootstrap/continuous_affine_full_v1",
        "previous": REPO / "data/refit_bootstrap/centered_poisson_obsrange_linf_refit_full_v1",
        "loader": "run_files.run_continuous_affine_full",
    },
    "eta_inf": {
        "draft": REPO / "data/refit_bootstrap/continuous_affine_eta_inf_v1",
        "previous": REPO / "data/refit_bootstrap/centered_poisson_obsrange_linf_refit_eta_inf_v1",
        "loader": "run_files.run_continuous_affine_eta_inf",
    },
}
OUT = REPO / "data/refit_bootstrap/three_tests_obsrange_v1"
BOOTSTRAPS = 1000
ALPHA = 0.05
SEED = 1729
TOLERANCE = 1e-12
VERSION = "three-tests-obsrange-v1"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "run_files"))
os.chdir(REPO)

from affine_poisson_multiplier_ks import _AffinePrepared, _prepare_sample, _validate_sample, affine_ref
from poisson_multiplier_ks import _linearized_process


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(scores3, weights3, scores4, weights4, L: float, U: float) -> _AffinePrepared:
    """Affine endpoints on [L, U]; 3b scores must lie in [L, U], 4b may lie outside."""

    scores3, weights3 = _validate_sample(scores3, weights3, name="3b")
    scores4 = np.ascontiguousarray(np.asarray(scores4, dtype=np.float64))
    weights4 = np.ascontiguousarray(np.asarray(weights4, dtype=np.float64))
    if not (np.all(np.isfinite(scores4)) and np.all(np.isfinite(weights4))):
        raise ValueError("4b scores and weights must be finite")
    if not np.sum(weights4) > 0:
        raise ValueError("4b weights must have a strictly positive total")
    if np.any(scores3 < L) or np.any(scores3 > U):
        raise ValueError("3b scores must lie within [L, U]")
    base3 = weights3 / np.sum(weights3)
    x = (scores3 - L) / (U - L)
    plus, minus = base3 * x, base3 * (1.0 - x)
    if plus.sum() <= 0 or minus.sum() <= 0:
        raise ValueError("both affine endpoint weight totals must be positive")
    support = np.unique(np.concatenate((scores3, scores4)))
    sample_plus = _prepare_sample(scores3, plus, support)
    sample_minus = _prepare_sample(scores3, minus, support)
    sample_4 = _prepare_sample(scores4, weights4, support)
    return _AffinePrepared(
        sample_plus=sample_plus,
        sample_minus=sample_minus,
        sample_4=sample_4,
        observed_base=sample_minus.cdf - sample_4.cdf,
        observed_direction=sample_plus.cdf - sample_minus.cdf,
        identity_t=float(plus.sum()),
        endpoint_total_plus=float(plus.sum()),
        endpoint_total_minus=float(minus.sum()),
    )


def three_tests(prepared: _AffinePrepared, draw) -> dict:
    """draw(replicate) -> (counts3, counts4)."""

    base = prepared.observed_base
    d_hat = prepared.observed_direction
    observed_envelope = affine_ref._norm_envelope(base, d_hat)
    statistic, t_hat = affine_ref._envelope_minimum(observed_envelope)
    statistic = float(statistic)

    fixed = np.empty(BOOTSTRAPS)
    refit = np.empty(BOOTSTRAPS)
    refit_t = np.empty(BOOTSTRAPS)
    intervals = []
    for replicate in range(BOOTSTRAPS):
        counts3, counts4 = draw(replicate)
        g_plus = _linearized_process(counts3, prepared.sample_plus)
        g_minus = _linearized_process(counts3, prepared.sample_minus)
        g4 = _linearized_process(counts4, prepared.sample_4)
        g_base = g_minus - g4
        g_dir = g_plus - g_minus
        intervals.extend(
            affine_ref._exceedance_intervals(observed_envelope, affine_ref._norm_envelope(g_base, g_dir), TOLERANCE)
        )
        fixed[replicate] = np.max(np.abs(g_base + t_hat * g_dir))
        refit[replicate], refit_t[replicate] = affine_ref._envelope_minimum(
            affine_ref._norm_envelope(g_base - t_hat * d_hat, d_hat + g_dir)
        )

    composite_count, composite_t = affine_ref._max_overlap(intervals)

    def p_of(count: int) -> float:
        return float((1 + count) / (BOOTSTRAPS + 1))

    p_composite = p_of(int(composite_count))
    p_fixed = p_of(int(np.count_nonzero(fixed + TOLERANCE >= statistic)))
    p_refit = p_of(int(np.count_nonzero(refit + TOLERANCE >= statistic)))
    side = "t=0" if t_hat <= TOLERANCE else ("t=1" if t_hat >= 1 - TOLERANCE else "interior")
    return {
        "statistic": statistic,
        "t_hat": float(t_hat),
        "t_hat_side": side,
        "composite": {"p_value": p_composite, "reject": p_composite <= ALPHA, "maximizing_t": float(composite_t)},
        "fixed": {"p_value": p_fixed, "reject": p_fixed <= ALPHA, "bootstrap_statistics": fixed},
        "refit": {
            "p_value": p_refit,
            "reject": p_refit <= ALPHA,
            "bootstrap_statistics": refit,
            "bootstrap_boundary_rate": float(np.mean((refit_t <= TOLERANCE) | (refit_t >= 1 - TOLERANCE))),
        },
    }


def class_stream_draw(n3: int, n4: int):
    seed3, seed4 = np.random.SeedSequence(SEED).spawn(2)
    rng3 = np.random.default_rng(seed3)
    rng4 = np.random.default_rng(seed4)
    return lambda _replicate: (rng3.poisson(1.0, size=n3), rng4.poisson(1.0, size=n4))


def run_one(row: dict, grid: dict, load) -> dict:
    started = time.perf_counter()
    _tinfo, scores3, weights3, scores4, weights4, clipped3, clipped4, cut = load(row)
    loaded = time.perf_counter()
    lower, upper = float(np.min(scores3)), float(np.max(scores3))
    prepared = prepare(scores3, weights3, scores4, weights4, lower, upper)
    result = three_tests(prepared, class_stream_draw(prepared.sample_plus.scores.size, prepared.sample_4.scores.size))
    with (grid["draft"] / "results" / f"{row['hash']}.pkl").open("rb") as handle:
        draft = pickle.load(handle)
    with (grid["previous"] / "results" / f"{row['hash']}.pkl").open("rb") as handle:
        previous = pickle.load(handle)
    return {
        **row,
        "version": VERSION,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "alpha": ALPHA,
        "score_clip_lower": cut,
        "score_clip_upper": 10.0,
        "range_lower": lower,
        "range_upper": upper,
        "n3": int(scores3.size),
        "n4": int(scores4.size),
        "n_clipped_3b": clipped3,
        "n_clipped_4b": clipped4,
        "has_signed_4b": bool(np.any(np.asarray(weights4) < 0)),
        **result,
        "previous_refit_p": float(previous["linf_literal"]["p_value"]),
        "draft_p_cut_to_10": float(draft["p_value"]),
        "load_seconds": loaded - started,
        "test_seconds": time.perf_counter() - loaded,
        "runner_sha256": sha256(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=sorted(GRIDS), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    grid = GRIDS[args.grid]
    load = __import__(grid["loader"], fromlist=["load_arrays_and_cutoff"]).load_arrays_and_cutoff
    with (grid["draft"] / "manifest.pkl").open("rb") as handle:
        rows = pickle.load(handle)
    selected = rows[args.shard_index :: args.n_shards]
    if args.limit is not None:
        selected = selected[: args.limit]
    out = OUT / args.grid / "results"
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"grid": args.grid, "selected": len(selected), "total": len(rows)}), flush=True)
    for position, row in enumerate(selected, start=1):
        destination = out / f"{row['hash']}.pkl"
        if destination.exists():
            with destination.open("rb") as handle:
                if pickle.load(handle).get("version") != VERSION:
                    raise RuntimeError(f"incompatible checkpoint {destination}")
            print(f"skip {position}/{len(selected)} {row['hash']}", flush=True)
            continue
        result = run_one(row, grid, load)
        temporary = destination.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(result, handle)
        os.replace(temporary, destination)
        print(
            json.dumps(
                {
                    "position": position,
                    "hash": row["hash"],
                    "exp": row["experiment_name"],
                    "eta": row["noise_scale"],
                    "eps": row["signal_ratio"],
                    "sr": row["sr_size"],
                    "seed": row["seed"],
                    "p_composite": result["composite"]["p_value"],
                    "p_fixed": result["fixed"]["p_value"],
                    "p_refit": result["refit"]["p_value"],
                    "p_refit_previous": result["previous_refit_p"],
                    "test_s": round(result["test_seconds"], 1),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
