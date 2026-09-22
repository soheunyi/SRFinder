"""A content-addressed store for SR statistics.

``signal_region.compute_sr_stats`` reads 15 step-2 records *and* their 15
step-1 encoders to produce two float32 arrays:

    ~315 MB of pickle reads, ~5 s  ->  8.4 MB of output

and throws the result away. A step-3 group of 100 recomputes it 100 times, and
every downstream analysis recomputes it again.

The key is ``(SR_stats_hashes, signal_filename, ensemble_mode, stats_type,
use_logits)``. Note what is *not* in it: ``s_SR``. The four SR sizes in the
campaign grid share one SR-stats array, so the store is reused four times over
before any analysis touches it.

Sizing: 2,900 distinct entries for the live manuscript (~24 GB), ~8,900 for the
whole cache (~75 GB).

This is Phase 4's ``artifacts_v2`` idea scoped to the one artifact that is
demonstrably hot. Nothing is migrated or deleted, ``compute_sr_stats`` is not
edited, and the store is only ever a cache: a miss recomputes, so correctness
never depends on it.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from typing import Iterable, Literal

import numpy as np

SOHEUN = pathlib.Path(__file__).resolve().parent.parent
STORE = SOHEUN / "data" / "artifacts_v2" / "sr_stats"
KEY_VERSION = 1


def sr_stats_key(
    hashes: Iterable[str],
    signal_filename: str,
    ensemble_mode: str = "max",
    stats_type: str = "smeared",
    use_logits: bool = True,
) -> str:
    """Content-derived key.

    The hashes are sorted: ``compute_sr_stats`` combines the ensemble with an
    element-wise max or mean, both order-independent, so the same ensemble
    listed in a different order is the same artifact.
    """
    payload = json.dumps(
        {
            "v": KEY_VERSION,
            "hashes": sorted(str(h) for h in hashes),
            "signal_filename": str(signal_filename),
            "ensemble_mode": str(ensemble_mode),
            "stats_type": str(stats_type),
            "use_logits": bool(use_logits),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


def _atomic_savez(path: pathlib.Path, **arrays: np.ndarray) -> None:
    """Write via a uniquely named temporary file and rename.

    The unique name matters: several slurm jobs can compute the same key at
    once, and a shared temporary path would let them clobber each other
    mid-write. ``os.replace`` is atomic, so the last writer simply wins with
    byte-identical content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    tmp_path = pathlib.Path(tmp)
    try:
        with open(tmp_path, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_or_compute_sr_stats(
    hashes: list[str],
    signal_filename: str,
    ensemble_mode: Literal["mean", "max"] = "max",
    stats_type: Literal["fvt", "smeared"] = "smeared",
    use_logits: bool = True,
    store: pathlib.Path = STORE,
    write: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Cached ``compute_sr_stats``. A miss computes and stores; a hit reads
    8.4 MB instead of 315 MB. dtypes are preserved exactly, so a hit returns
    what a miss would have returned."""
    from signal_region import compute_sr_stats as _compute

    key = sr_stats_key(hashes, signal_filename, ensemble_mode, stats_type, use_logits)
    path = pathlib.Path(store) / f"{key}.npz"

    if path.exists():
        try:
            with np.load(path) as z:
                return z["train"], z["tst"]
        except Exception:  # noqa: BLE001 - a truncated file must not be fatal
            pass

    train, tst = _compute(
        hashes, signal_filename, ensemble_mode, stats_type, use_logits
    )
    if write:
        _atomic_savez(path, train=train, tst=tst)
        meta = {
            "key": key,
            "key_version": KEY_VERSION,
            "hashes": sorted(str(h) for h in hashes),
            "signal_filename": str(signal_filename),
            "ensemble_mode": str(ensemble_mode),
            "stats_type": str(stats_type),
            "use_logits": bool(use_logits),
            "train_shape": list(train.shape),
            "tst_shape": list(tst.shape),
            "dtype": str(train.dtype),
        }
        mpath = path.with_suffix(".json")
        fd, tmp = tempfile.mkstemp(dir=str(mpath.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        os.replace(tmp, mpath)
    return train, tst


def install(store: pathlib.Path = STORE, write: bool = True):
    """Route every ``compute_sr_stats`` call through the store.

    Patches two namespaces. ``step_3_preprocessing`` does
    ``from signal_region import compute_sr_stats``, so it holds its own
    reference and patching ``signal_region`` alone would miss it.

    Returns a callable that undoes the patch.
    """
    import signal_region

    originals = {}

    def cached(hashes, signal_filename, ensemble_mode="max",
               stats_type="smeared", use_logits=True):
        return load_or_compute_sr_stats(
            hashes, signal_filename, ensemble_mode, stats_type, use_logits,
            store=store, write=write,
        )

    originals["signal_region"] = signal_region.compute_sr_stats
    signal_region.compute_sr_stats = cached

    try:
        import step_3_preprocessing

        originals["step_3_preprocessing"] = step_3_preprocessing.compute_sr_stats
        step_3_preprocessing.compute_sr_stats = cached
    except ImportError:
        pass

    def uninstall() -> None:
        signal_region.compute_sr_stats = originals["signal_region"]
        if "step_3_preprocessing" in originals:
            import step_3_preprocessing

            step_3_preprocessing.compute_sr_stats = originals["step_3_preprocessing"]

    return uninstall


def store_stats(store: pathlib.Path = STORE) -> dict:
    store = pathlib.Path(store)
    if not store.exists():
        return {"entries": 0, "bytes": 0}
    files = list(store.glob("*.npz"))
    return {
        "entries": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "path": str(store),
    }
