"""Exact-equivalence and invariance tests for the signed-4b extension."""
from dataclasses import asdict

import numpy as np

import affine_weighted_ks_compiled as nonnegative
import affine_weighted_ks_signed_compiled as fast
import affine_weighted_ks_signed_reference as slow


rng = np.random.default_rng(9173)

# The signed extension must be exactly identical to the frozen nonnegative
# implementation whenever all weights are nonnegative.
for case in range(100):
    n3 = int(rng.integers(10, 100))
    n4 = int(rng.integers(10, 100))
    z3 = rng.uniform(size=n3)
    z4 = rng.uniform(size=n4)
    if case % 2 == 0:
        z3 = np.round(z3, 1)
        z4 = np.round(z4, 1)
    w3 = rng.lognormal(size=n3)
    w4 = rng.lognormal(size=n4)
    kwargs = dict(L=0.0, U=1.0, B=29, seed=case)
    expected = nonnegative.affine_ks_test(z3, w3, z4, w4, **kwargs)
    observed = fast.affine_ks_test(z3, w3, z4, w4, **kwargs)
    assert asdict(expected) == asdict(observed), (case, expected, observed)

# The Python and compiled-envelope signed implementations must agree exactly.
for case in range(100):
    n3 = int(rng.integers(10, 100))
    n4 = int(rng.integers(10, 100))
    z3 = rng.uniform(size=n3)
    z4 = rng.uniform(size=n4)
    w3 = rng.lognormal(size=n3)
    w4 = rng.lognormal(size=n4)
    negative = rng.choice(n4, size=max(1, n4 // 8), replace=False)
    w4[negative] *= -0.2
    assert w4.sum() > 0
    kwargs = dict(L=0.0, U=1.0, B=29, seed=case)
    expected = slow.affine_ks_test(z3, w3, z4, w4, **kwargs)
    observed = fast.affine_ks_test(z3, w3, z4, w4, **kwargs)
    assert asdict(expected) == asdict(observed), (case, expected, observed)

    # Positive class-wise rescaling must cancel exactly.
    scaled = fast.affine_ks_test(z3, 7.0 * w3, z4, 11.0 * w4, **kwargs)
    for field in (
        "p_value",
        "reject",
        "max_exceedances",
        "bootstrap_replicates",
        "n3",
        "n4",
        "distinct_scores",
        "numerical_tolerance",
    ):
        assert getattr(observed, field) == getattr(scaled, field), (
            case,
            field,
            observed,
            scaled,
        )
    for field in ("alpha", "ks_statistic", "ks_t", "maximizing_p_t"):
        assert np.isclose(
            getattr(observed, field),
            getattr(scaled, field),
            rtol=1e-14,
            atol=1e-15,
        ), (case, field, observed, scaled)
    for diagnostic in ("effective_sample_sizes", "max_normalized_weights"):
        assert all(
            np.isclose(
                getattr(observed, diagnostic)[key],
                getattr(scaled, diagnostic)[key],
                rtol=1e-14,
                atol=0.0,
            )
            for key in getattr(observed, diagnostic)
        ), (case, diagnostic, observed, scaled)

try:
    fast.affine_ks_test(
        [0.2, 0.8], [1.0, 1.0], [0.3, 0.7], [-2.0, 1.0],
        L=0.0, U=1.0, B=9, seed=1,
    )
except ValueError as error:
    assert "strictly positive total" in str(error)
else:
    raise AssertionError("A nonpositive signed 4b total was accepted")

print(
    "100 nonnegative exact-equivalence, 100 signed compiled-equivalence, "
    "and signed scaling/validation checks passed",
    flush=True,
)
