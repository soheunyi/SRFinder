from typing import Literal
import warnings
import numpy as np
from tqdm import tqdm
from ks_test import max_cdf_diff
from joblib import Parallel, delayed


def cdf_bootstrap_1d(
    empirical_cdf: np.ndarray,
    stats: np.ndarray,
    n_samples: int,
    random_seed: None | int = None,
):
    assert np.all(stats[1:] >= stats[:-1]), "stats must be sorted"

    if random_seed is not None:
        np.random.seed(random_seed)

    unif_samples = np.random.uniform(0, 1, size=n_samples)

    # Binary search to find the first index where empirical_cdf >= unif_samples
    bootstrap_idx = np.searchsorted(empirical_cdf, unif_samples)
    bootstrap_idx = np.clip(bootstrap_idx, 0, len(stats) - 1)

    return stats[bootstrap_idx]


def _null_bootstrap_worker(
    empirical_cdf_pool: np.ndarray,
    stats_pool_sorted: np.ndarray,
    n_samples_1: int,
    n_samples_2: int,
    seed: int,
):
    np.random.seed(seed)
    stats_bst_1 = cdf_bootstrap_1d(empirical_cdf_pool, stats_pool_sorted, n_samples_1)
    stats_bst_2 = cdf_bootstrap_1d(empirical_cdf_pool, stats_pool_sorted, n_samples_2)

    return max_cdf_diff(
        stats_bst_1,
        stats_bst_2,
        np.ones_like(stats_bst_1),
        np.ones_like(stats_bst_2),
    )


def null_max_cdf_diff_bootstrap(
    stats_1: np.ndarray,
    stats_2: np.ndarray,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
    n_samples_1: int,
    n_samples_2: int,
    n_reps: int,
    random_seed: int = 0,
    do_tqdm: bool = True,
    n_jobs: int = -1,
):
    stats_pool = np.concatenate([stats_1, stats_2])
    weights_pool = np.concatenate([weights_1, weights_2])
    stats_pool_sorted_idx = np.argsort(stats_pool)
    stats_pool_sorted = stats_pool[stats_pool_sorted_idx]
    weights_pool_sorted = weights_pool[stats_pool_sorted_idx]
    empirical_cdf_pool = np.cumsum(weights_pool_sorted) / np.sum(weights_pool_sorted)

    # Generate seeds for each worker
    np.random.seed(random_seed)
    seeds = np.random.randint(0, 2**32, size=n_reps)

    # Parallel processing
    max_cdf_diffs = Parallel(n_jobs=n_jobs)(
        delayed(_null_bootstrap_worker)(
            empirical_cdf_pool,
            stats_pool_sorted,
            n_samples_1,
            n_samples_2,
            seed,
        )
        for seed in tqdm(seeds, disable=not do_tqdm)
    )

    return np.array(max_cdf_diffs)


def _alt_bootstrap_worker(
    empirical_cdf_1: np.ndarray,
    stats_1_sorted: np.ndarray,
    empirical_cdf_2: np.ndarray,
    stats_2_sorted: np.ndarray,
    n_samples_1: int,
    n_samples_2: int,
    seed: int,
):
    np.random.seed(seed)
    stats_bst_1 = cdf_bootstrap_1d(empirical_cdf_1, stats_1_sorted, n_samples_1)
    stats_bst_2 = cdf_bootstrap_1d(empirical_cdf_2, stats_2_sorted, n_samples_2)

    return max_cdf_diff(
        stats_bst_1,
        stats_bst_2,
        np.ones_like(stats_bst_1),
        np.ones_like(stats_bst_2),
    )


def alt_max_cdf_diff_bootstrap(
    stats_1: np.ndarray,
    stats_2: np.ndarray,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
    n_samples_1: int,
    n_samples_2: int,
    n_reps: int,
    random_seed: int = 0,
    do_tqdm: bool = True,
    n_jobs: int = -1,
):
    stats_1_sorted_idx = np.argsort(stats_1)
    stats_1_sorted = stats_1[stats_1_sorted_idx]
    weights_1_sorted = weights_1[stats_1_sorted_idx]
    empirical_cdf_1 = np.cumsum(weights_1_sorted) / np.sum(weights_1_sorted)

    stats_2_sorted_idx = np.argsort(stats_2)
    stats_2_sorted = stats_2[stats_2_sorted_idx]
    weights_2_sorted = weights_2[stats_2_sorted_idx]
    empirical_cdf_2 = np.cumsum(weights_2_sorted) / np.sum(weights_2_sorted)

    # Generate seeds for each worker
    np.random.seed(random_seed)
    seeds = np.random.randint(0, 2**32, size=n_reps)

    # Parallel processing
    max_cdf_diffs = Parallel(n_jobs=n_jobs)(
        delayed(_alt_bootstrap_worker)(
            empirical_cdf_1,
            stats_1_sorted,
            empirical_cdf_2,
            stats_2_sorted,
            n_samples_1,
            n_samples_2,
            seed,
        )
        for seed in tqdm(seeds, disable=not do_tqdm)
    )

    return np.array(max_cdf_diffs)


def _null_bootstrap_worker_poisson(
    stats_pool: np.ndarray,
    weights_pool: np.ndarray,
    n_samples_1: int,
    n_samples_2: int,
    seed: int,
):
    np.random.seed(seed)
    poisson_samples = np.random.poisson(lam=1, size=n_samples_1)
    return stats_pool[poisson_samples]
