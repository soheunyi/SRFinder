"""Replicate-level equivalence tests for the C++ multiplier kernel."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from poisson_multiplier_ks import (
    bootstrap_statistics_from_multipliers,
    prepare_weighted_ks,
    weighted_ks_test,
)
from poisson_multiplier_ks_compiled import (
    bootstrap_statistics_from_multipliers_compiled,
)


def main():
    rng = np.random.default_rng(20260921)
    for case in range(50):
        n3 = int(rng.integers(3, 80))
        n4 = int(rng.integers(3, 80))
        scores_3 = np.round(rng.normal(size=n3), decimals=1)
        scores_4 = np.round(rng.normal(size=n4), decimals=1)
        weights_3 = rng.uniform(0.01, 3.0, size=n3)
        weights_4 = rng.uniform(0.01, 3.0, size=n4)
        _, sample_3, sample_4 = prepare_weighted_ks(
            scores_3,
            weights_3,
            scores_4,
            weights_4,
        )
        counts_3 = rng.poisson(1.0, size=(37, n3))
        counts_4 = rng.poisson(1.0, size=(37, n4))
        for form in ("linearized", "direct_normalized"):
            expected = bootstrap_statistics_from_multipliers(
                sample_3,
                sample_4,
                counts_3,
                counts_4,
                form=form,
            )
            found = bootstrap_statistics_from_multipliers_compiled(
                sample_3,
                sample_4,
                counts_3,
                counts_4,
                form=form,
            )
            assert np.allclose(found, expected, atol=2e-15, equal_nan=True), (
                case,
                form,
                np.nanmax(np.abs(found - expected)),
            )
    python_result = weighted_ks_test(
        scores_3,
        weights_3,
        scores_4,
        weights_4,
        bootstrap_replicates=101,
        seed=17,
        multiplier_chunk_size=13,
        implementation="python",
    )
    compiled_result = weighted_ks_test(
        scores_3,
        weights_3,
        scores_4,
        weights_4,
        bootstrap_replicates=101,
        seed=17,
        multiplier_chunk_size=13,
        implementation="cpp",
    )
    assert python_result.p_value == compiled_result.p_value
    assert np.allclose(
        python_result.bootstrap_statistics,
        compiled_result.bootstrap_statistics,
        atol=2e-15,
    )
    print("Passed 50 random Python/C++ equivalence cases for both bootstrap forms.")


if __name__ == "__main__":
    main()
