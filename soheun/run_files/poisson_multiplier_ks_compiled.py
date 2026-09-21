"""ctypes adapter for the centered Poisson weighted-KS C++ kernel."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
CODE_ROOT = ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

from poisson_multiplier_ks import BootstrapForm, _PreparedSample


LIBRARY = ctypes.CDLL(str(ROOT / "poisson_multiplier_ks_kernel.so"))
array_float64 = np.ctypeslib.ndpointer(
    dtype=np.float64,
    ndim=1,
    flags="C_CONTIGUOUS",
)
array_int64 = np.ctypeslib.ndpointer(
    dtype=np.int64,
    ndim=1,
    flags="C_CONTIGUOUS",
)
LIBRARY.poisson_multiplier_ks_batch.argtypes = [
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    array_int64,
    array_float64,
    array_float64,
    array_int64,
    array_float64,
    array_float64,
    array_int64,
    array_int64,
    ctypes.c_int,
    array_float64,
]
LIBRARY.poisson_multiplier_ks_batch.restype = ctypes.c_int


def bootstrap_statistics_from_multipliers_compiled(
    sample_3: _PreparedSample,
    sample_4: _PreparedSample,
    multipliers_3,
    multipliers_4,
    *,
    form: BootstrapForm = "linearized",
) -> np.ndarray:
    counts_3 = np.ascontiguousarray(multipliers_3, dtype=np.int64)
    counts_4 = np.ascontiguousarray(multipliers_4, dtype=np.int64)
    if counts_3.ndim != 2 or counts_4.ndim != 2:
        raise ValueError("multiplier arrays must be two-dimensional")
    if counts_3.shape[0] != counts_4.shape[0]:
        raise ValueError("3b and 4b multiplier arrays must have equal replicates")
    if counts_3.shape[1] != sample_3.scores.size:
        raise ValueError("3b multiplier width does not match sample size")
    if counts_4.shape[1] != sample_4.scores.size:
        raise ValueError("4b multiplier width does not match sample size")
    if np.any(counts_3 < 0) or np.any(counts_4 < 0):
        raise ValueError("multipliers must be nonnegative")
    if form not in {"linearized", "direct_normalized"}:
        raise ValueError(f"unknown bootstrap form: {form}")

    result = np.empty(counts_3.shape[0], dtype=np.float64)
    return_code = LIBRARY.poisson_multiplier_ks_batch(
        sample_3.cdf.size,
        sample_3.scores.size,
        sample_4.scores.size,
        counts_3.shape[0],
        np.ascontiguousarray(sample_3.support_indices, dtype=np.int64),
        np.ascontiguousarray(sample_3.normalized_weights, dtype=np.float64),
        np.ascontiguousarray(sample_3.cdf, dtype=np.float64),
        np.ascontiguousarray(sample_4.support_indices, dtype=np.int64),
        np.ascontiguousarray(sample_4.normalized_weights, dtype=np.float64),
        np.ascontiguousarray(sample_4.cdf, dtype=np.float64),
        counts_3.reshape(-1),
        counts_4.reshape(-1),
        int(form == "direct_normalized"),
        result,
    )
    if return_code:
        raise RuntimeError(f"C++ Poisson multiplier kernel failed with {return_code}")
    return result
