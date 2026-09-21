"""Mathematical and regression tests for poisson_multiplier_ks.py."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poisson_multiplier_ks import (
    bootstrap_statistics_from_multipliers,
    draw_poisson_multipliers,
    prepare_weighted_ks,
    weighted_ks_test,
)


def _slow_cdf(scores, weights, support):
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    return np.array(
        [np.sum(weights[scores <= point]) for point in support],
        dtype=float,
    )


def _slow_linearized(scores, normalized_weights, support, cdf, counts):
    return np.array(
        [
            np.sum(
                (counts - 1)
                * normalized_weights
                * ((scores <= point) - cdf[index])
            )
            for index, point in enumerate(support)
        ]
    )


def test_preparation_and_ties():
    support, sample_3, sample_4 = prepare_weighted_ks(
        np.array([0.0, 0.5, 0.5, 1.0]),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0.25, 0.5, 0.75]),
        np.array([2.0, 1.0, 1.0]),
    )
    assert np.array_equal(support, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(
        sample_3.cdf,
        _slow_cdf(sample_3.scores, sample_3.normalized_weights, support),
    )
    assert np.allclose(
        sample_4.cdf,
        _slow_cdf(sample_4.scores, sample_4.normalized_weights, support),
    )


def test_supplied_multipliers_against_direct_calculation():
    support, sample_3, sample_4 = prepare_weighted_ks(
        np.array([-1.0, 0.0, 2.0]),
        np.array([1.0, 2.0, 1.0]),
        np.array([-0.5, 1.0]),
        np.array([3.0, 1.0]),
    )
    counts_3 = np.array([[2, 0, 1], [0, 1, 3]])
    counts_4 = np.array([[1, 2], [4, 0]])
    found = bootstrap_statistics_from_multipliers(
        sample_3,
        sample_4,
        counts_3,
        counts_4,
        form="linearized",
    )
    expected = []
    for c3, c4 in zip(counts_3, counts_4):
        g3 = _slow_linearized(
            sample_3.scores,
            sample_3.normalized_weights,
            support,
            sample_3.cdf,
            c3,
        )
        g4 = _slow_linearized(
            sample_4.scores,
            sample_4.normalized_weights,
            support,
            sample_4.cdf,
            c4,
        )
        expected.append(np.max(np.abs(g3 - g4)))
    assert np.allclose(found, expected, atol=1e-15)


def test_direct_normalized_against_direct_calculation():
    support, sample_3, sample_4 = prepare_weighted_ks(
        np.array([-1.0, 0.0, 2.0]),
        np.array([1.0, 2.0, 1.0]),
        np.array([-0.5, 1.0]),
        np.array([3.0, 1.0]),
    )
    counts_3 = np.array([[2, 0, 1], [0, 1, 3]])
    counts_4 = np.array([[1, 2], [4, 0]])
    found = bootstrap_statistics_from_multipliers(
        sample_3,
        sample_4,
        counts_3,
        counts_4,
        form="direct_normalized",
    )
    expected = []
    for c3, c4 in zip(counts_3, counts_4):
        f3 = _slow_cdf(
            sample_3.scores,
            c3 * sample_3.normalized_weights,
            support,
        )
        f4 = _slow_cdf(
            sample_4.scores,
            c4 * sample_4.normalized_weights,
            support,
        )
        expected.append(
            np.max(np.abs((f3 - sample_3.cdf) - (f4 - sample_4.cdf)))
        )
    assert np.allclose(found, expected, atol=1e-15)


def test_scale_invariance_and_reproducibility():
    rng = np.random.default_rng(12)
    scores_3 = rng.normal(size=31)
    scores_4 = rng.normal(size=27)
    weights_3 = rng.uniform(0.1, 2.0, size=31)
    weights_4 = rng.uniform(0.1, 2.0, size=27)
    result = weighted_ks_test(
        scores_3,
        weights_3,
        scores_4,
        weights_4,
        bootstrap_replicates=29,
        seed=91,
        multiplier_chunk_size=7,
    )
    repeat = weighted_ks_test(
        scores_3,
        weights_3 * 800,
        scores_4,
        weights_4 * 0.002,
        bootstrap_replicates=29,
        seed=91,
        multiplier_chunk_size=7,
    )
    assert np.isclose(result.statistic, repeat.statistic, atol=1e-15)
    assert result.p_value == repeat.p_value
    assert np.allclose(
        result.bootstrap_statistics,
        repeat.bootstrap_statistics,
        atol=1e-15,
    )
    different_chunking = weighted_ks_test(
        scores_3,
        weights_3,
        scores_4,
        weights_4,
        bootstrap_replicates=29,
        seed=91,
        multiplier_chunk_size=19,
    )
    assert np.array_equal(
        result.bootstrap_statistics,
        different_chunking.bootstrap_statistics,
    )


def test_multiplier_generation_is_class_specific():
    rng = np.random.default_rng(1729)
    counts_3, counts_4 = draw_poisson_multipliers(
        replicates=2000,
        n3=3,
        n4=4,
        rng=rng,
    )
    assert counts_3.shape == (2000, 3)
    assert counts_4.shape == (2000, 4)
    assert abs(np.mean(counts_3) - 1.0) < 0.04
    assert abs(np.var(counts_3) - 1.0) < 0.08
    assert abs(np.corrcoef(counts_3[:, 0], counts_4[:, 0])[0, 1]) < 0.08
    assert np.any(counts_3 >= 2), "regression guard: multiplicities must not be Boolean"


def test_validation():
    try:
        weighted_ks_test([0], [-1], [0], [1], bootstrap_replicates=2)
    except ValueError as error:
        assert "nonnegative" in str(error)
    else:
        raise AssertionError("negative weights should fail")

    try:
        weighted_ks_test([0], [0], [0], [1], bootstrap_replicates=2)
    except ValueError as error:
        assert "positive finite total" in str(error)
    else:
        raise AssertionError("zero total weight should fail")


def main():
    test_preparation_and_ties()
    test_supplied_multipliers_against_direct_calculation()
    test_direct_normalized_against_direct_calculation()
    test_scale_invariance_and_reproducibility()
    test_multiplier_generation_is_class_specific()
    test_validation()
    print(
        "Passed weighted-CDF, supplied-multiplier, direct-normalization, "
        "scale-invariance, reproducibility, independence, and validation checks."
    )


if __name__ == "__main__":
    main()
