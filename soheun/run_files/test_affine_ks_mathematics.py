"""Independent mathematical checks for the continuous affine KS test.

These checks exercise the Python reference implementation directly.  They are
deliberately separate from test_affine_ks_compiled.py, which checks exact
equivalence of the compiled envelope kernel to the reference implementation.
"""

import numpy as np

import affine_weighted_ks_reference as affine


def evaluate_envelope(envelope, points):
    indices = np.clip(
        np.searchsorted(envelope.left, points, side="right") - 1,
        0,
        len(envelope.left) - 1,
    )
    return envelope.intercept[indices] + envelope.slope[indices] * points


def test_interval_edge_cases():
    observed = affine._norm_envelope(np.array([0.5]), np.array([0.0]))
    bootstrap = affine._norm_envelope(np.array([-1.0]), np.array([2.0]))
    assert np.allclose(
        affine._exceedance_intervals(observed, bootstrap, 0.0),
        [(0.0, 0.25), (0.75, 1.0)],
    )

    observed_point = affine._norm_envelope(np.array([-0.5]), np.array([1.0]))
    bootstrap_zero = affine._norm_envelope(np.array([0.0]), np.array([0.0]))
    assert np.allclose(
        affine._exceedance_intervals(observed_point, bootstrap_zero, 0.0),
        [(0.5, 0.5)],
    )
    assert affine._max_overlap([(0.0, 0.5), (0.5, 1.0), (0.5, 0.5)])[0] == 3


def test_random_envelopes(rng, cases=200):
    grid = np.linspace(0.0, 1.0, 2001)
    for case in range(cases):
        slopes = rng.normal(size=30)
        intercepts = rng.normal(size=30)
        if case % 3 == 0:
            slopes = np.round(slopes)
        envelope = affine._upper_envelope(slopes, intercepts)
        truth = np.max(
            intercepts[:, None] + slopes[:, None] * grid,
            axis=0,
        )
        assert np.allclose(truth, evaluate_envelope(envelope, grid), atol=1e-12)

        other_slopes = rng.normal(size=20)
        other_intercepts = rng.normal(size=20)
        other = affine._upper_envelope(other_slopes, other_intercepts)
        intervals = affine._exceedance_intervals(envelope, other, 0.0)
        expected = np.max(
            other_intercepts[:, None] + other_slopes[:, None] * grid,
            axis=0,
        ) >= truth
        found = np.zeros(len(grid), dtype=bool)
        for lower, upper in intervals:
            found |= (grid >= lower) & (grid <= upper)
        assert np.array_equal(found, expected)


def test_end_to_end_against_direct_process(rng, cases=50):
    grid = np.linspace(0.0, 1.0, 2001)
    for case in range(cases):
        scores_3b = rng.integers(0, 11, size=12) / 10
        scores_4b = rng.integers(0, 11, size=15) / 10
        weights_3b = rng.uniform(0.1, 2.0, len(scores_3b))
        weights_4b = rng.uniform(0.1, 2.0, len(scores_4b))
        result = affine.affine_ks_test(
            scores_3b,
            weights_3b,
            scores_4b,
            weights_4b,
            L=0.0,
            U=1.0,
            B=19,
            seed=case,
        )
        rescaled = affine.affine_ks_test(
            scores_3b,
            weights_3b * 700,
            scores_4b,
            weights_4b * 0.003,
            L=0.0,
            U=1.0,
            B=19,
            seed=case,
        )
        assert result.p_value == rescaled.p_value
        assert abs(result.ks_statistic - rescaled.ks_statistic) < 1e-12

        order_3b = np.argsort(scores_3b, kind="stable")
        order_4b = np.argsort(scores_4b, kind="stable")
        x = scores_3b[order_3b]
        y = scores_4b[order_4b]
        w = weights_3b[order_3b]
        r = weights_4b[order_4b] / np.sum(weights_4b)
        plus = w * x
        plus /= np.sum(plus)
        minus = w * (1 - x)
        minus /= np.sum(minus)
        support = np.unique(np.r_[x, y])
        indicators_3b = x[:, None] <= support
        indicators_4b = y[:, None] <= support
        cdf_plus = plus @ indicators_3b
        cdf_minus = minus @ indicators_3b
        cdf_4b = r @ indicators_4b

        mixture_values = np.r_[grid, result.maximizing_p_t]
        observed = np.max(
            np.abs(
                (cdf_minus - cdf_4b)[:, None]
                + (cdf_plus - cdf_minus)[:, None] * mixture_values
            ),
            axis=0,
        )
        direct_rng = np.random.default_rng(case)
        exceedances = np.zeros(len(mixture_values), dtype=int)
        for _ in range(19):
            xi = direct_rng.poisson(1.0, len(x)) - 1
            zeta = direct_rng.poisson(1.0, len(y)) - 1
            process_plus = (xi * plus) @ (indicators_3b - cdf_plus)
            process_minus = (xi * minus) @ (indicators_3b - cdf_minus)
            process_4b = (zeta * r) @ (indicators_4b - cdf_4b)
            bootstrap = np.max(
                np.abs(
                    (process_minus - process_4b)[:, None]
                    + (process_plus - process_minus)[:, None] * mixture_values
                ),
                axis=0,
            )
            exceedances += (
                bootstrap + result.numerical_tolerance + 1e-13 >= observed
            )
        assert exceedances[:-1].max() <= result.max_exceedances
        assert exceedances[-1] >= result.max_exceedances
        assert result.ks_statistic <= observed[:-1].min() + 1e-12


def main():
    rng = np.random.default_rng(1742)
    test_interval_edge_cases()
    test_random_envelopes(rng)
    test_end_to_end_against_direct_process(rng)
    print(
        "Passed interval edge cases, 200 random envelope checks, and "
        "50 direct end-to-end multiplier-bootstrap checks."
    )


if __name__ == "__main__":
    main()
