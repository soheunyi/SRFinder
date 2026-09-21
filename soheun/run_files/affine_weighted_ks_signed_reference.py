"""Signed-4b extension of the continuous affine-nuisance multiplier KS test.

The original implementation in ``affine_weighted_ks_reference.py`` is left
unchanged.  This module changes only validation and normalization of the 4b
Monte Carlo weights: 3b weights must still be nonnegative, while 4b weights
may be signed provided that their total is strictly positive.

For signed 4b weights, the normalized cumulative sum is a Monte Carlo
estimator of a physical CDF but need not itself be monotone.  The centered
multiplier process applies to the signed influence contributions

    r_j (1{Z_j <= y} - F_hat(y)),  r_j = w_j / sum_k w_k.

This is a first-order, conditional approximation for independent event
records.  Its use requires a positive limiting total weight, a nonnegative
limiting physical measure, suitable weight moments, and no dominating
absolute normalized weight.  It does not turn signed weights into
probabilities and never truncates, takes absolute values of, or drops an
event weight.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
from numpy.typing import ArrayLike, NDArray


ROOT = Path(__file__).resolve().parent
_NAME = "_private_signed_affine_base_reference"
_SPEC = importlib.util.spec_from_file_location(
    _NAME, ROOT / "affine_weighted_ks_reference.py"
)
ref = importlib.util.module_from_spec(_SPEC)
sys.modules[_NAME] = ref
_SPEC.loader.exec_module(ref)

AffineKSResult = ref.AffineKSResult
_Envelope = ref._Envelope
_validate_nonnegative_sample = ref._validate_sample


def _validate_sample(
    scores: ArrayLike, weights: ArrayLike, name: str, L: float, U: float
) -> tuple[NDArray, NDArray]:
    if name != "4b":
        return _validate_nonnegative_sample(scores, weights, name, L, U)

    z = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if z.ndim != 1 or w.ndim != 1 or z.size != w.size or z.size == 0:
        raise ValueError(
            f"{name}: scores and weights must be nonempty 1D arrays of equal size."
        )
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(w)):
        raise ValueError(
            f"{name}: all scores and weights must be finite; handle clipping beforehand."
        )
    if np.any(z < L) or np.any(z > U):
        raise ValueError(
            f"{name}: scores must already lie within the specified support [L, U]."
        )
    scale = float(np.max(np.abs(w)))
    if not scale > 0:
        raise ValueError(f"{name}: at least one weight must be nonzero.")
    if not float(np.sum(w / scale, dtype=np.float64)) > 0:
        raise ValueError(f"{name}: signed weights must have strictly positive total.")
    order = np.argsort(z, kind="stable")
    return z[order], w[order] / scale


# The base function resolves this global at call time.  All statistic,
# multiplier, envelope, tolerance, and p-value code remains byte-for-byte the
# validated base implementation.
ref._validate_sample = _validate_sample


def affine_ks_test(*args, **kwargs) -> AffineKSResult:
    result = ref.affine_ks_test(*args, **kwargs)
    z4 = np.asarray(args[2], dtype=np.float64)
    w4 = np.asarray(args[3], dtype=np.float64)
    order = np.argsort(z4, kind="stable")
    scaled = w4[order] / np.max(np.abs(w4))
    normalized = scaled / np.sum(scaled, dtype=np.float64)
    maxima = dict(result.max_normalized_weights)
    maxima["4b"] = float(np.max(np.abs(normalized)))
    return replace(result, max_normalized_weights=maxima)
