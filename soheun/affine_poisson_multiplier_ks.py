"""Fixed and composite affine corrections with centered Poisson multipliers.

Both procedures use the same affine family and the same class-specific
Poisson(1) multiplier streams as :mod:`poisson_multiplier_ks`.  The fixed
procedure chooses the affine correction by minimizing mean absolute CDF
discrepancy on the observed sample.  The composite procedure maximizes the
pointwise multiplier p-value over the full continuous affine family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from poisson_multiplier_ks import (
    _PreparedSample,
    _linearized_process,
    _prepare_sample,
    _validate_multipliers,
    _validate_sample,
)
from run_files import affine_weighted_ks_compiled as affine


affine_ref = affine.ref


CorrectionMode = Literal["fixed", "composite_supremum"]


@dataclass(frozen=True)
class AffineMultiplierKSResult:
    p_value: float
    reject: bool
    alpha: float
    statistic: float
    correction_mode: CorrectionMode
    fitted_t: float
    identity_t: float
    maximizing_p_t: float
    max_exceedances: int
    bootstrap_replicates: int
    seed: int | None
    n3: int
    n4: int
    distinct_scores: int
    correction_value_L: float
    correction_value_U: float
    fit_objective: float
    bootstrap_statistics: np.ndarray | None
    effective_sample_sizes: dict[str, float]
    max_normalized_weights: dict[str, float]
    numerical_tolerance: float
    multiplier_stream: str


@dataclass(frozen=True)
class _AffinePrepared:
    sample_plus: _PreparedSample
    sample_minus: _PreparedSample
    sample_4: _PreparedSample
    observed_base: np.ndarray
    observed_direction: np.ndarray
    identity_t: float
    endpoint_total_plus: float
    endpoint_total_minus: float


def _prepare_affine(
    scores_3,
    weights_3,
    scores_4,
    weights_4,
    *,
    L: float,
    U: float,
) -> _AffinePrepared:
    scores_3, weights_3 = _validate_sample(scores_3, weights_3, name="3b")
    scores_4, weights_4 = _validate_sample(scores_4, weights_4, name="4b")
    if not np.isfinite(L) or not np.isfinite(U) or not L < U:
        raise ValueError("Require finite L < U")
    if np.any(scores_3 < L) or np.any(scores_3 > U):
        raise ValueError("3b scores must lie within [L, U]")
    if np.any(scores_4 < L) or np.any(scores_4 > U):
        raise ValueError("4b scores must lie within [L, U]")

    base_3 = weights_3 / np.sum(weights_3)
    x = (scores_3 - L) / (U - L)
    endpoint_plus = base_3 * x
    endpoint_minus = base_3 * (1.0 - x)
    total_plus = float(np.sum(endpoint_plus))
    total_minus = float(np.sum(endpoint_minus))
    if total_plus <= 0 or total_minus <= 0:
        raise ValueError("both affine endpoint weight totals must be positive")

    support = np.unique(np.concatenate((scores_3, scores_4)))
    sample_plus = _prepare_sample(scores_3, endpoint_plus, support)
    sample_minus = _prepare_sample(scores_3, endpoint_minus, support)
    sample_4 = _prepare_sample(scores_4, weights_4, support)
    return _AffinePrepared(
        sample_plus=sample_plus,
        sample_minus=sample_minus,
        sample_4=sample_4,
        observed_base=sample_minus.cdf - sample_4.cdf,
        observed_direction=sample_plus.cdf - sample_minus.cdf,
        identity_t=total_plus,
        endpoint_total_plus=total_plus,
        endpoint_total_minus=total_minus,
    )


def fit_mean_absolute_t(base: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
    """Exactly minimize mean(abs(base + t * direction)) over t in [0, 1]."""

    nonzero = direction != 0
    if not np.any(nonzero):
        t = 0.5
    else:
        # |a_i + d_i t| = |d_i| |t - (-a_i/d_i)|.  An unconstrained
        # minimizer is therefore a weighted median of the roots with weights
        # |d_i|; projection onto [0, 1] gives the constrained minimizer.
        roots = -base[nonzero] / direction[nonzero]
        weights = np.abs(direction[nonzero])
        order = np.argsort(roots, kind="stable")
        ordered_roots = roots[order]
        cumulative = np.cumsum(weights[order])
        index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
        t = float(np.clip(ordered_roots[index], 0.0, 1.0))
    objective = float(np.mean(np.abs(base + t * direction)))
    return t, objective


def _correction_endpoints(prepared: _AffinePrepared, t: float) -> tuple[float, float]:
    # The base 3b weights integrate this affine correction to one.  Identity is
    # therefore exactly (h(L), h(U)) = (1, 1) at t=identity_t.
    return (
        float((1.0 - t) / prepared.endpoint_total_minus),
        float(t / prepared.endpoint_total_plus),
    )


def _validate_supplied_multipliers(
    multipliers,
    *,
    B: int,
    n3: int,
    n4: int,
) -> tuple[np.ndarray, np.ndarray]:
    if multipliers is None:
        raise ValueError("internal error: missing multipliers")
    counts_3, counts_4 = multipliers
    return (
        _validate_multipliers(counts_3, replicates=B, sample_size=n3, name="3b"),
        _validate_multipliers(counts_4, replicates=B, sample_size=n4, name="4b"),
    )


def affine_multiplier_ks_test(
    scores_3,
    weights_3,
    scores_4,
    weights_4,
    *,
    L: float,
    U: float,
    correction_mode: CorrectionMode,
    bootstrap_replicates: int = 1000,
    alpha: float = 0.05,
    seed: int | None = 0,
    numerical_tolerance: float = 1e-12,
    multipliers: tuple[np.ndarray, np.ndarray] | None = None,
) -> AffineMultiplierKSResult:
    """Run a fixed-fit or composite affine centered-multiplier KS test."""

    if correction_mode not in {"fixed", "composite_supremum"}:
        raise ValueError(f"unknown correction mode: {correction_mode}")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if not np.isfinite(numerical_tolerance) or numerical_tolerance < 0:
        raise ValueError("numerical_tolerance must be finite and nonnegative")

    prepared = _prepare_affine(
        scores_3,
        weights_3,
        scores_4,
        weights_4,
        L=L,
        U=U,
    )
    n3 = prepared.sample_plus.scores.size
    n4 = prepared.sample_4.scores.size
    fitted_t, fit_objective = fit_mean_absolute_t(
        prepared.observed_base,
        prepared.observed_direction,
    )
    correction_L, correction_U = _correction_endpoints(prepared, fitted_t)
    observed_envelope = affine_ref._norm_envelope(
        prepared.observed_base,
        prepared.observed_direction,
    )

    if correction_mode == "fixed":
        statistic = float(
            np.max(
                np.abs(
                    prepared.observed_base
                    + fitted_t * prepared.observed_direction
                )
            )
        )
        fitted_ks_t = fitted_t
        bootstrap_statistics = np.empty(bootstrap_replicates, dtype=np.float64)
        intervals = None
    else:
        statistic, fitted_ks_t = affine_ref._envelope_minimum(observed_envelope)
        bootstrap_statistics = None
        intervals = []

    if multipliers is None:
        seed_sequence = np.random.SeedSequence(seed)
        seed_3, seed_4 = seed_sequence.spawn(2)
        rng_3 = np.random.default_rng(seed_3)
        rng_4 = np.random.default_rng(seed_4)
        supplied_3 = supplied_4 = None
    else:
        supplied_3, supplied_4 = _validate_supplied_multipliers(
            multipliers,
            B=bootstrap_replicates,
            n3=n3,
            n4=n4,
        )
        rng_3 = rng_4 = None

    for replicate in range(bootstrap_replicates):
        if multipliers is None:
            counts_3 = rng_3.poisson(1.0, size=n3)
            counts_4 = rng_4.poisson(1.0, size=n4)
        else:
            counts_3 = supplied_3[replicate]
            counts_4 = supplied_4[replicate]
        process_plus = _linearized_process(counts_3, prepared.sample_plus)
        process_minus = _linearized_process(counts_3, prepared.sample_minus)
        process_4 = _linearized_process(counts_4, prepared.sample_4)
        bootstrap_base = process_minus - process_4
        bootstrap_direction = process_plus - process_minus

        if correction_mode == "fixed":
            bootstrap_statistics[replicate] = np.max(
                np.abs(bootstrap_base + fitted_t * bootstrap_direction)
            )
        else:
            bootstrap_envelope = affine_ref._norm_envelope(
                bootstrap_base,
                bootstrap_direction,
            )
            intervals.extend(
                affine_ref._exceedance_intervals(
                    observed_envelope,
                    bootstrap_envelope,
                    numerical_tolerance,
                )
            )

    if correction_mode == "fixed":
        exceedances = int(
            np.count_nonzero(
                bootstrap_statistics + numerical_tolerance >= statistic
            )
        )
        maximizing_p_t = fitted_t
    else:
        exceedances, maximizing_p_t = affine_ref._max_overlap(intervals)
    p_value = float((1 + exceedances) / (bootstrap_replicates + 1))

    endpoint_weights = {
        "3b_plus": prepared.sample_plus.normalized_weights,
        "3b_minus": prepared.sample_minus.normalized_weights,
        "4b": prepared.sample_4.normalized_weights,
    }
    return AffineMultiplierKSResult(
        p_value=p_value,
        reject=bool(p_value <= alpha),
        alpha=float(alpha),
        statistic=float(statistic),
        correction_mode=correction_mode,
        fitted_t=float(fitted_ks_t),
        identity_t=float(prepared.identity_t),
        maximizing_p_t=float(maximizing_p_t),
        max_exceedances=exceedances,
        bootstrap_replicates=int(bootstrap_replicates),
        seed=seed,
        n3=int(n3),
        n4=int(n4),
        distinct_scores=int(prepared.sample_4.cdf.size),
        correction_value_L=correction_L,
        correction_value_U=correction_U,
        fit_objective=fit_objective,
        bootstrap_statistics=bootstrap_statistics,
        effective_sample_sizes={
            name: float(1.0 / np.sum(weights**2))
            for name, weights in endpoint_weights.items()
        },
        max_normalized_weights={
            name: float(np.max(weights))
            for name, weights in endpoint_weights.items()
        },
        numerical_tolerance=float(numerical_tolerance),
        multiplier_stream="independent_class_seedsequence_spawn_v1",
    )
