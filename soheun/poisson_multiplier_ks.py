"""Centered Poisson(1) multiplier bootstrap for weighted two-sample KS tests.

The test implemented here keeps the two samples separate.  Conditional on the
observed score--weight pairs, independent Poisson(1) multipliers approximate
the two weighted empirical-process fluctuations.  The bootstrap statistic is
therefore centered; it is not the raw KS distance between two independently
reweighted empirical distributions.

This module is the readable reference implementation.  Production kernels may
accelerate it, but should accept the same multiplier arrays and agree with it
replicate by replicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


BootstrapForm = Literal["linearized", "direct_normalized"]
Implementation = Literal["python", "cpp"]


@dataclass(frozen=True)
class WeightedKSTestResult:
    """Result of a centered Poisson multiplier weighted-KS test."""

    statistic: float
    p_value: float
    reject: bool
    bootstrap_statistics: np.ndarray
    bootstrap_form: BootstrapForm
    implementation: Implementation
    bootstrap_replicates: int
    alpha: float
    seed: int | None
    n3: int
    n4: int
    effective_sample_size_3: float
    effective_sample_size_4: float
    max_normalized_weight_3: float
    max_normalized_weight_4: float


@dataclass(frozen=True)
class _PreparedSample:
    scores: np.ndarray
    normalized_weights: np.ndarray
    support_indices: np.ndarray
    cdf: np.ndarray


def _as_float_vector(values, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if result.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _validate_sample(scores, weights, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    scores_array = _as_float_vector(scores, name=f"{name} scores")
    weights_array = _as_float_vector(weights, name=f"{name} weights")
    if scores_array.size != weights_array.size:
        raise ValueError(f"{name} scores and weights must have the same length")
    if np.any(weights_array < 0):
        raise ValueError(f"{name} weights must be nonnegative")
    total = float(np.sum(weights_array))
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"{name} weights must have a positive finite total")
    return scores_array, weights_array


def _prepare_sample(
    scores: np.ndarray,
    weights: np.ndarray,
    support: np.ndarray,
) -> _PreparedSample:
    normalized = weights / np.sum(weights)
    support_indices = np.searchsorted(support, scores)
    masses = np.bincount(
        support_indices,
        weights=normalized,
        minlength=support.size,
    )
    return _PreparedSample(
        scores=scores,
        normalized_weights=np.ascontiguousarray(normalized),
        support_indices=np.ascontiguousarray(support_indices),
        cdf=np.cumsum(masses),
    )


def prepare_weighted_ks(
    scores_3,
    weights_3,
    scores_4,
    weights_4,
) -> tuple[np.ndarray, _PreparedSample, _PreparedSample]:
    """Validate inputs and place both weighted CDFs on their union support."""

    scores_3, weights_3 = _validate_sample(scores_3, weights_3, name="3b")
    scores_4, weights_4 = _validate_sample(scores_4, weights_4, name="4b")
    support = np.unique(np.concatenate((scores_3, scores_4)))
    return (
        support,
        _prepare_sample(scores_3, weights_3, support),
        _prepare_sample(scores_4, weights_4, support),
    )


def draw_poisson_multipliers(
    *,
    replicates: int,
    n3: int,
    n4: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw independent class-specific Poisson(1) multiplier arrays."""

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if n3 <= 0 or n4 <= 0:
        raise ValueError("sample sizes must be positive")
    return (
        rng.poisson(1.0, size=(replicates, n3)),
        rng.poisson(1.0, size=(replicates, n4)),
    )


def _validate_multipliers(
    multipliers,
    *,
    replicates: int,
    sample_size: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(multipliers)
    if result.shape != (replicates, sample_size):
        raise ValueError(
            f"{name} multipliers must have shape {(replicates, sample_size)}"
        )
    if not np.issubdtype(result.dtype, np.integer):
        if not np.all(np.equal(result, np.floor(result))):
            raise ValueError(f"{name} multipliers must be integer-valued")
        result = result.astype(np.int64)
    if np.any(result < 0):
        raise ValueError(f"{name} multipliers must be nonnegative")
    return np.ascontiguousarray(result)


def _linearized_process(
    counts: np.ndarray,
    sample: _PreparedSample,
) -> np.ndarray:
    centered_weight = (counts - 1.0) * sample.normalized_weights
    masses = np.bincount(
        sample.support_indices,
        weights=centered_weight,
        minlength=sample.cdf.size,
    )
    return np.cumsum(masses) - sample.cdf * np.sum(centered_weight)


def _direct_normalized_process(
    counts: np.ndarray,
    sample: _PreparedSample,
) -> np.ndarray | None:
    replicate_weights = counts * sample.normalized_weights
    total = float(np.sum(replicate_weights))
    if total <= 0 or not np.isfinite(total):
        return None
    masses = np.bincount(
        sample.support_indices,
        weights=replicate_weights,
        minlength=sample.cdf.size,
    )
    return np.cumsum(masses) / total - sample.cdf


def bootstrap_statistics_from_multipliers(
    sample_3: _PreparedSample,
    sample_4: _PreparedSample,
    multipliers_3,
    multipliers_4,
    *,
    form: BootstrapForm = "linearized",
) -> np.ndarray:
    """Compute centered weighted-KS statistics for supplied multipliers.

    Supplying the multipliers explicitly makes comparisons between bootstrap
    forms and future compiled kernels exactly paired.
    """

    multipliers_3 = np.asarray(multipliers_3)
    multipliers_4 = np.asarray(multipliers_4)
    if multipliers_3.ndim != 2 or multipliers_4.ndim != 2:
        raise ValueError("multiplier arrays must be two-dimensional")
    if multipliers_3.shape[0] != multipliers_4.shape[0]:
        raise ValueError("3b and 4b multiplier arrays must have equal replicates")
    replicates = multipliers_3.shape[0]
    multipliers_3 = _validate_multipliers(
        multipliers_3,
        replicates=replicates,
        sample_size=sample_3.scores.size,
        name="3b",
    )
    multipliers_4 = _validate_multipliers(
        multipliers_4,
        replicates=replicates,
        sample_size=sample_4.scores.size,
        name="4b",
    )
    if form not in {"linearized", "direct_normalized"}:
        raise ValueError(f"unknown bootstrap form: {form}")

    result = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        if form == "linearized":
            process_3 = _linearized_process(multipliers_3[replicate], sample_3)
            process_4 = _linearized_process(multipliers_4[replicate], sample_4)
        else:
            process_3 = _direct_normalized_process(
                multipliers_3[replicate], sample_3
            )
            process_4 = _direct_normalized_process(
                multipliers_4[replicate], sample_4
            )
            if process_3 is None or process_4 is None:
                result[replicate] = np.nan
                continue
        result[replicate] = np.max(np.abs(process_3 - process_4))
    return result


def weighted_ks_test(
    scores_3,
    weights_3,
    scores_4,
    weights_4,
    *,
    bootstrap_replicates: int = 1000,
    alpha: float = 0.05,
    seed: int | None = 0,
    form: BootstrapForm = "linearized",
    multiplier_chunk_size: int = 64,
    multipliers: tuple[np.ndarray, np.ndarray] | None = None,
    implementation: Implementation = "python",
) -> WeightedKSTestResult:
    """Test equality of two weighted score distributions.

    The returned p-value compares the observed weighted KS distance with a
    centered Poisson(1) multiplier approximation to its null law.
    """

    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if multiplier_chunk_size <= 0:
        raise ValueError("multiplier_chunk_size must be positive")
    if implementation not in {"python", "cpp"}:
        raise ValueError(f"unknown implementation: {implementation}")

    _, sample_3, sample_4 = prepare_weighted_ks(
        scores_3,
        weights_3,
        scores_4,
        weights_4,
    )
    observed = float(np.max(np.abs(sample_3.cdf - sample_4.cdf)))

    seed_sequence = np.random.SeedSequence(seed)
    class_seed_3, class_seed_4 = seed_sequence.spawn(2)
    rng_3 = np.random.default_rng(class_seed_3)
    rng_4 = np.random.default_rng(class_seed_4)
    statistics = np.empty(bootstrap_replicates, dtype=np.float64)
    if implementation == "python":
        statistic_function = bootstrap_statistics_from_multipliers
    else:
        from run_files.poisson_multiplier_ks_compiled import (
            bootstrap_statistics_from_multipliers_compiled,
        )

        statistic_function = bootstrap_statistics_from_multipliers_compiled
    if multipliers is not None:
        multiplier_3, multiplier_4 = multipliers
        multiplier_3 = _validate_multipliers(
            multiplier_3,
            replicates=bootstrap_replicates,
            sample_size=sample_3.scores.size,
            name="3b",
        )
        multiplier_4 = _validate_multipliers(
            multiplier_4,
            replicates=bootstrap_replicates,
            sample_size=sample_4.scores.size,
            name="4b",
        )

    for start in range(0, bootstrap_replicates, multiplier_chunk_size):
        stop = min(start + multiplier_chunk_size, bootstrap_replicates)
        if multipliers is None:
            counts_3 = rng_3.poisson(
                1.0,
                size=(stop - start, sample_3.scores.size),
            )
            counts_4 = rng_4.poisson(
                1.0,
                size=(stop - start, sample_4.scores.size),
            )
        else:
            counts_3 = multiplier_3[start:stop]
            counts_4 = multiplier_4[start:stop]
        statistics[start:stop] = statistic_function(
            sample_3,
            sample_4,
            counts_3,
            counts_4,
            form=form,
        )

    finite = np.isfinite(statistics)
    if not np.any(finite):
        raise RuntimeError("all bootstrap replicates had invalid total weight")
    # A direct-normalized replicate can have zero total weight.  Such a draw is
    # undefined and is excluded explicitly rather than silently assigned zero.
    valid_statistics = statistics[finite]
    p_value = float(
        (1 + np.count_nonzero(valid_statistics >= observed))
        / (valid_statistics.size + 1)
    )

    ess_3 = float(1.0 / np.sum(sample_3.normalized_weights**2))
    ess_4 = float(1.0 / np.sum(sample_4.normalized_weights**2))
    return WeightedKSTestResult(
        statistic=observed,
        p_value=p_value,
        reject=p_value <= alpha,
        bootstrap_statistics=statistics,
        bootstrap_form=form,
        implementation=implementation,
        bootstrap_replicates=bootstrap_replicates,
        alpha=alpha,
        seed=seed,
        n3=sample_3.scores.size,
        n4=sample_4.scores.size,
        effective_sample_size_3=ess_3,
        effective_sample_size_4=ess_4,
        max_normalized_weight_3=float(np.max(sample_3.normalized_weights)),
        max_normalized_weight_4=float(np.max(sample_4.normalized_weights)),
    )
