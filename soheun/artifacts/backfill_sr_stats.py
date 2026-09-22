"""Populate the SR-stats store, and verify it is faithful.

    # verify the store returns exactly what recomputing returns
    python artifacts/backfill_sr_stats.py --verify --limit 5

    # fill the configs the phase-5 benchmark uses
    python artifacts/backfill_sr_stats.py --pattern 'configs/tmp/CR_fvt_training_ensemble_max_*_0_0.1_0.2_0.8.yml'

    # fill everything reachable from the step-3 configs on disk
    python artifacts/backfill_sr_stats.py --workers 12
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = pathlib.Path(__file__).resolve().parent
SOHEUN = HERE.parent
for p in (str(SOHEUN), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(SOHEUN)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sr_stats import (  # noqa: E402
    STORE,
    load_or_compute_sr_stats,
    sr_stats_key,
    store_stats,
)

DEFAULT_PATTERN = "configs/tmp/CR_fvt_training_ensemble_max_*.yml"


def spec_of(path: str) -> tuple | None:
    """The (hashes, signal, mode, type) a step-3 config implies, or None."""
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        sr = cfg["signal_region"]
        return (
            tuple(sr["SR_stats_hashes"]),
            cfg["dataset"]["signal_filename"],
            sr.get("ensemble_mode", "max"),
            sr.get("stats_type", "smeared"),
        )
    except Exception:  # noqa: BLE001
        return None


def build(spec) -> tuple[str, float, bool]:
    hashes, signal, mode, stype = spec
    key = sr_stats_key(hashes, signal, mode, stype, True)
    path = pathlib.Path(STORE) / f"{key}.npz"
    if path.exists():
        return key, 0.0, True
    t0 = time.perf_counter()
    load_or_compute_sr_stats(list(hashes), signal, mode, stype, True)
    return key, time.perf_counter() - t0, False


def verify(spec) -> tuple[str, bool, str]:
    """A hit must return exactly what a miss would have returned."""
    from signal_region import compute_sr_stats

    hashes, signal, mode, stype = spec
    a_tr, a_ts = load_or_compute_sr_stats(list(hashes), signal, mode, stype, True)
    b_tr, b_ts = compute_sr_stats(list(hashes), signal, mode, stype, True)
    key = sr_stats_key(hashes, signal, mode, stype, True)
    if a_tr.dtype != b_tr.dtype:
        return key, False, f"dtype {a_tr.dtype} vs {b_tr.dtype}"
    if not (np.array_equal(a_tr, b_tr) and np.array_equal(a_ts, b_ts)):
        return key, False, "values differ"
    return key, True, f"exact, {a_tr.shape[0]:,} + {a_ts.shape[0]:,} rows"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=DEFAULT_PATTERN)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(SOHEUN / args.pattern)))
    specs = {}
    for p in paths:
        s = spec_of(p)
        if s is not None:
            specs[s] = None
    todo = list(specs)
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(paths):,} configs -> {len(specs):,} distinct SR-stats specs")
    print(f"store: {store_stats()}")
    if not todo:
        return 0

    if args.verify:
        ok = True
        for s in todo:
            key, good, detail = verify(s)
            print(f"  {'PASS' if good else 'FAIL'}  {key[:16]}  {detail}")
            ok = ok and good
        print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    t0 = time.perf_counter()
    built = hits = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(build, s): s for s in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                _, _, was_hit = fut.result()
                built += not was_hit
                hits += was_hit
                if i % 50 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  built={built} hit={hits}  "
                          f"{time.perf_counter()-t0:.0f}s", flush=True)
    else:
        for i, s in enumerate(todo, 1):
            _, _, was_hit = build(s)
            built += not was_hit
            hits += was_hit
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  built={built} hit={hits}  "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)

    st = store_stats()
    print(f"\ndone in {time.perf_counter()-t0:.0f}s  built={built} hit={hits}")
    print(f"store: {st['entries']:,} entries, {st['bytes']/1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
