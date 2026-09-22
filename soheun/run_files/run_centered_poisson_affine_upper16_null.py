"""Run fixed/composite affine tests with support [SR threshold L,16]."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import sys
import time


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
DATA_REPO = Path(
    os.environ.get("SRFINDER_DATA_REPO", "/home/export/soheuny/SRFinder/soheun")
)
OUT = DATA_REPO / "data/refit_bootstrap/centered_poisson_affine_upper16_null_v1"
VERSION = "centered-poisson1-affine-upper16-null-v1"
SUPPORT_UPPER = 16.0

from run_files import run_centered_poisson_affine_symmetric16_null as base


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(row: dict) -> dict:
    from affine_poisson_multiplier_ks import affine_multiplier_ks_test
    from run_files.run_centered_poisson_affine_null import load_arrays_and_cutoff

    started = time.perf_counter()
    scores_3, weights_3, scores_4, weights_4, clipped_3, clipped_4, lower = (
        load_arrays_and_cutoff(row)
    )
    if clipped_3 or clipped_4:
        raise RuntimeError(
            f"{row['hash']}: source clipped scores at 10; cannot infer [L,16] input"
        )
    loaded = time.perf_counter()
    fixed = affine_multiplier_ks_test(
        scores_3,
        weights_3,
        scores_4,
        weights_4,
        L=lower,
        U=SUPPORT_UPPER,
        correction_mode="fixed",
        bootstrap_replicates=base.BOOTSTRAPS,
        alpha=base.ALPHA,
        seed=base.SEED,
        numerical_tolerance=base.NUMERICAL_TOLERANCE,
    )
    fixed_done = time.perf_counter()
    composite = affine_multiplier_ks_test(
        scores_3,
        weights_3,
        scores_4,
        weights_4,
        L=lower,
        U=SUPPORT_UPPER,
        correction_mode="composite_supremum",
        bootstrap_replicates=base.BOOTSTRAPS,
        alpha=base.ALPHA,
        seed=base.SEED,
        numerical_tolerance=base.NUMERICAL_TOLERANCE,
    )
    completed = time.perf_counter()
    return {
        **row,
        "version": VERSION,
        "bootstrap_scheme": "independent_centered_poisson1_multiplier",
        "multiplier_stream": "independent_class_seedsequence_spawn_v1",
        "bootstrap_seed": base.SEED,
        "bootstrap_replicates": base.BOOTSTRAPS,
        "alpha": base.ALPHA,
        "correction_support_lower": lower,
        "correction_support_upper": SUPPORT_UPPER,
        "n_clipped_3b": clipped_3,
        "n_clipped_4b": clipped_4,
        "local_support": base.load_local_result(row),
        "fixed": asdict(fixed),
        "composite": asdict(composite),
        "load_seconds": loaded - started,
        "fixed_seconds": fixed_done - loaded,
        "composite_seconds": completed - fixed_done,
        "module_sha256": file_sha256(CODE_ROOT / "affine_poisson_multiplier_ks.py"),
        "shared_module_sha256": file_sha256(CODE_ROOT / "poisson_multiplier_ks.py"),
        "base_runner_sha256": file_sha256(
            CODE_ROOT / "run_files/run_centered_poisson_affine_null.py"
        ),
        "runner_sha256": file_sha256(Path(__file__)),
    }


base.OUT = OUT
base.MANIFEST = OUT / "manifest.pkl"
base.VERSION = VERSION
base.run_one = run_one


if __name__ == "__main__":
    base.main()
