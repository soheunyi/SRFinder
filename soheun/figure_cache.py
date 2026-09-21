"""Disk cache for the expensive aggregation behind the manuscript figures.

The figure notebooks spend nearly all of their time loading TrainingInfo
records -- thousands of pickles per figure -- not drawing. Caching the
aggregated arrays means a style-only change (fonts, label sizes, figure width)
replots in seconds instead of re-reading the whole experiment tree.

Environment:
    FIGURE_CACHE=on       use the cache (default)
    FIGURE_CACHE=refresh  recompute and overwrite
    FIGURE_CACHE=off      ignore the cache entirely
"""
import hashlib
import json
import os
import pickle
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "data" / "figure_cache"


def _mode():
    return os.environ.get("FIGURE_CACHE", "on").strip().lower()


def key(name, *parts):
    """Stable cache key from a name plus the inputs the result depends on."""
    blob = json.dumps(parts, sort_keys=True, default=repr)
    return f"{name}__{hashlib.sha256(blob.encode()).hexdigest()[:16]}"


def load(cache_key):
    if _mode() in ("off", "refresh"):
        return None
    path = CACHE_DIR / f"{cache_key}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as handle:
            value = pickle.load(handle)
    except Exception as error:  # a corrupt cache must never fail a figure
        print(f"[figure_cache] ignoring {path.name}: {error}", flush=True)
        return None
    print(f"[figure_cache] hit {path.name}", flush=True)
    return value


def save(cache_key, value):
    if _mode() == "off":
        return value
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{cache_key}.pkl"
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(value, handle)
    tmp.replace(path)
    print(f"[figure_cache] stored {path.name}", flush=True)
    return value
