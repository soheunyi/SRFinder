"""Mathematical tests for fixed and composite affine multiplier procedures."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from affine_poisson_multiplier_ks import (
    _prepare_affine,
    affine_multiplier_ks_test,
    fit_mean_absolute_t,
)


def test_exact_mean_absolute_fit():
    rng = np.random.default_rng(82)
    for _ in range(100):
        base = rng.normal(size=30)
        direction = rng.normal(size=30)
        found_t, found_value = fit_mean_absolute_t(base, direction)
        grid = np.linspace(0.0, 1.0, 20001)
        grid_values = np.mean(
            np.abs(base[:, None] + direction[:, None] * grid),
            axis=0,
        )
        assert found_value <= np.min(grid_values) + 1e-12
        assert 0 <= found_t <= 1


def test_identity_parameter_and_correction_endpoints():
    prepared = _prepare_affine(
        np.array([0.1, 0.4, 0.8]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.2, 0.7]),
        np.array([2.0, 1.0]),
        L=0.0,
        U=1.0,
    )
    t = prepared.identity_t
    mixed = t * prepared.sample_plus.cdf + (1 - t) * prepared.sample_minus.cdf
    base_weights = np.array([1.0, 2.0, 3.0])
    base_weights /= base_weights.sum()
    expected = np.array(
        [np.sum(base_weights[np.array([0.1, 0.4, 0.8]) <= y])
         for y in np.unique(np.r_[0.1, 0.4, 0.8, 0.2, 0.7])]
    )
    assert np.allclose(mixed, expected, atol=1e-15)


def test_fixed_and_composite_with_supplied_multipliers():
    rng = np.random.default_rng(103)
    scores_3 = rng.uniform(size=23)
    scores_4 = rng.uniform(size=19)
    weights_3 = rng.lognormal(size=23)
    weights_4 = rng.lognormal(size=19)
    counts = (
        rng.poisson(1.0, size=(59, len(scores_3))),
        rng.poisson(1.0, size=(59, len(scores_4))),
    )
    fixed = affine_multiplier_ks_test(
        scores_3,
        weights_3,
        scores_4,
        weights_4,
        L=0.0,
        U=1.0,
        correction_mode="fixed",
        bootstrap_replicates=59,
        multipliers=counts,
    )
    composite = affine_multiplier_ks_test(
        scores_3,
        weights_3,
        scores_4,
        weights_4,
        L=0.0,
        U=1.0,
        correction_mode="composite_supremum",
        bootstrap_replicates=59,
        multipliers=counts,
    )
    fixed_exceedances = int(
        np.count_nonzero(
            fixed.bootstrap_statistics + fixed.numerical_tolerance
            >= fixed.statistic
        )
    )
    assert fixed.max_exceedances == fixed_exceedances
    # The composite procedure maximizes pointwise exceedance counts, so it
    # cannot have fewer exceedances than the fitted fixed correction.
    assert composite.max_exceedances >= fixed.max_exceedances
    assert composite.p_value >= fixed.p_value
    assert composite.bootstrap_statistics is None

    scaled = affine_multiplier_ks_test(
        scores_3,
        weights_3 * 70,
        scores_4,
        weights_4 * 0.04,
        L=0.0,
        U=1.0,
        correction_mode="composite_supremum",
        bootstrap_replicates=59,
        multipliers=counts,
    )
    assert composite.p_value == scaled.p_value
    assert np.isclose(composite.statistic, scaled.statistic, atol=1e-15)


def test_seed_reproducibility():
    rng = np.random.default_rng(9)
    args = (
        rng.uniform(size=31),
        rng.uniform(0.1, 2.0, size=31),
        rng.uniform(size=27),
        rng.uniform(0.1, 2.0, size=27),
    )
    first = affine_multiplier_ks_test(
        *args,
        L=0.0,
        U=1.0,
        correction_mode="fixed",
        bootstrap_replicates=41,
        seed=77,
    )
    second = affine_multiplier_ks_test(
        *args,
        L=0.0,
        U=1.0,
        correction_mode="fixed",
        bootstrap_replicates=41,
        seed=77,
    )
    assert first.p_value == second.p_value
    assert np.array_equal(first.bootstrap_statistics, second.bootstrap_statistics)


def main():
    test_exact_mean_absolute_fit()
    test_identity_parameter_and_correction_endpoints()
    test_fixed_and_composite_with_supplied_multipliers()
    test_seed_reproducibility()
    print(
        "Passed exact affine fitting, identity mapping, supplied-multiplier, "
        "composite-dominance, scale-invariance, and reproducibility checks."
    )


if __name__ == "__main__":
    main()
