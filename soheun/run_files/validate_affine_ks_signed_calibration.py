"""Development null-calibration check for the signed-4b extension."""
import json

import numpy as np

from affine_weighted_ks_signed_compiled import affine_ks_test


REPETITIONS = 100
BOOTSTRAPS = 199
N3 = 500
N4 = 500
ALPHA = 0.05
rng = np.random.default_rng(20260919)
rejections = 0
p_values = []

for repetition in range(REPETITIONS):
    z3 = rng.uniform(size=N3)
    z4 = rng.uniform(size=N4)
    w3 = np.ones(N3)
    # Weights are independent of the score, so their signed weighted limiting
    # distribution remains Uniform(0,1).  The negative absolute-weight share is
    # about 2%, slightly above the largest representative real ZH value.
    w4 = np.where(rng.random(N4) < 0.1, -0.2, 1.0)
    result = affine_ks_test(
        z3,
        w3,
        z4,
        w4,
        L=0.0,
        U=1.0,
        B=BOOTSTRAPS,
        alpha=ALPHA,
        seed=10000 + repetition,
        numerical_tol=1e-12,
    )
    p_values.append(result.p_value)
    rejections += int(result.reject)

rate = rejections / REPETITIONS
summary = {
    "repetitions": REPETITIONS,
    "bootstrap_replicates": BOOTSTRAPS,
    "n3": N3,
    "n4": N4,
    "alpha": ALPHA,
    "rejections": rejections,
    "rejection_rate": rate,
    "mean_p_value": float(np.mean(p_values)),
    "negative_abs_fraction_target": 0.1 * 0.2 / (0.9 + 0.1 * 0.2),
}
print(json.dumps(summary, indent=2), flush=True)
if rate > 0.12:
    raise RuntimeError(f"Signed development null rejection rate is unexpectedly high: {rate}")
