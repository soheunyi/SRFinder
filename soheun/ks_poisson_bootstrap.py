from typing import Literal
import warnings
import numpy as np
from tqdm import tqdm
from ks_test import max_cdf_diff
from joblib import Parallel, delayed


def _null_bootstrap_worker(
    stats_3b: np.ndarray,
    stats_4b: np.ndarray,
    weights_3b: np.ndarray,
    weights_4b: np.ndarray,
    seed: int,
):
    np.random.seed(seed)
    stats_pooled = np.concatenate([stats_3b, stats_4b])
    weights_pooled = np.concatenate([weights_3b, weights_4b])

    total_weight_3b = np.sum(weights_3b)
    total_weight_4b = np.sum(weights_4b)
    poisson_samples_3b = np.random.poisson(
        lam=total_weight_3b / (total_weight_3b + total_weight_4b),
        size=len(stats_pooled),
    )
    poisson_samples_nz_3b = poisson_samples_3b > 0

    stats_pooled_nz_3b = stats_pooled[poisson_samples_nz_3b]
    weights_pooled_nz_3b = (
        weights_pooled[poisson_samples_nz_3b]
        * poisson_samples_nz_3b[poisson_samples_nz_3b]
    )

    poisson_samples_4b = np.random.poisson(
        lam=total_weight_4b / (total_weight_3b + total_weight_4b),
        size=len(stats_pooled),
    )
    poisson_samples_nz_4b = poisson_samples_4b > 0

    stats_pooled_nz_4b = stats_pooled[poisson_samples_nz_4b]
    weights_pooled_nz_4b = (
        weights_pooled[poisson_samples_nz_4b]
        * poisson_samples_nz_4b[poisson_samples_nz_4b]
    )

    return max_cdf_diff(
        stats_pooled_nz_3b,
        stats_pooled_nz_4b,
        weights_pooled_nz_3b,
        weights_pooled_nz_4b,
    )


def null_max_cdf_diff_bootstrap(
    stats_1: np.ndarray,
    stats_2: np.ndarray,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
    n_reps: int,
    random_seed: int = 0,
    do_tqdm: bool = True,
    n_jobs: int = -1,
):

    # Generate seeds for each worker
    np.random.seed(random_seed)
    seeds = np.random.randint(0, 2**32, size=n_reps)

    # Parallel processing
    max_cdf_diffs = Parallel(n_jobs=n_jobs)(
        delayed(_null_bootstrap_worker)(
            stats_1,
            stats_2,
            weights_1,
            weights_2,
            seed,
        )
        for seed in tqdm(seeds, disable=not do_tqdm)
    )

    return np.array(max_cdf_diffs)


def _alt_bootstrap_worker(
    stats_1: np.ndarray,
    stats_2: np.ndarray,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
    seed: int,
):
    np.random.seed(seed)
    poisson_samples_1 = np.random.poisson(lam=1, size=len(stats_1))
    poisson_samples_nz_1 = poisson_samples_1 > 0
    stats_1_nz = stats_1[poisson_samples_nz_1]
    weights_1_nz = (
        weights_1[poisson_samples_nz_1] * poisson_samples_nz_1[poisson_samples_nz_1]
    )

    poisson_samples_2 = np.random.poisson(lam=1, size=len(stats_2))
    poisson_samples_nz_2 = poisson_samples_2 > 0
    stats_2_nz = stats_2[poisson_samples_nz_2]
    weights_2_nz = (
        weights_2[poisson_samples_nz_2] * poisson_samples_nz_2[poisson_samples_nz_2]
    )

    return max_cdf_diff(
        stats_1_nz,
        stats_2_nz,
        weights_1_nz,
        weights_2_nz,
    )


def alt_max_cdf_diff_bootstrap(
    stats_1: np.ndarray,
    stats_2: np.ndarray,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
    n_reps: int,
    random_seed: int = 0,
    do_tqdm: bool = True,
    n_jobs: int = -1,
):
    # Generate seeds for each worker
    np.random.seed(random_seed)
    seeds = np.random.randint(0, 2**32, size=n_reps)

    # Parallel processing
    max_cdf_diffs = Parallel(n_jobs=n_jobs)(
        delayed(_alt_bootstrap_worker)(
            stats_1,
            stats_2,
            weights_1,
            weights_2,
            seed,
        )
        for seed in tqdm(seeds, disable=not do_tqdm)
    )

    return np.array(max_cdf_diffs)
