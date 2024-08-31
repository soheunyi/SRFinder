import numpy as np
import cudf
from cuml.neighbors import NearestNeighbors
import multiprocessing as mp

import tqdm

from events_data import EventsData


def dist_func(x, y):
    return np.linalg.norm(x - y, axis=1)


def min_dist_fn(i, X_sorted):
    if i == 0:
        return np.inf
    else:
        return np.min(dist_func(X_sorted[i], X_sorted[:i]))


def dist_to_nearest_peak(
    X: np.ndarray,
    rho: np.ndarray,
    weights: np.ndarray,
    peak_pct: float,
    nbd_pct: float = 0.02,
    n_points: int = 10000,
    seed: int = None,
    n_workers: int = 4,
):
    """
    Find center of peaks of rho with respect to X.
    Return a function that calculates (scaled) minimum distance to the nearest peak center.

    https://www.science.org/doi/10.1126/science.1242072

    Parameters
    ----------
    X : np.ndarray
        Data points.
    rho : np.ndarray
        Values of rho for each data point.
    weights : np.ndarray
        Weights for sampling.
    peak_pct : float
        Percentage of points to consider as peaks.
    nbd_pct : float
        Percentage of nearest neighbors used for calculating minimum distance scales.
    n_points : int
        Number of points to sample.
    seed : int, optional
        Random seed for reproducibility, by default None.
    """
    assert len(weights) == len(X) == len(rho)
    if seed is not None:
        np.random.seed(seed)
    n_points = min(n_points, X.shape[0])
    random_idx = np.random.choice(
        X.shape[0], n_points, replace=False, p=weights / np.sum(weights)
    )
    X = X[random_idx]

    n_neighbors = int(n_points * nbd_pct)
    X_cudf = cudf.DataFrame(X)
    model = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="auto", metric="euclidean"
    )
    nbrs = model.fit(X_cudf)
    distances, indices = nbrs.kneighbors(X_cudf)
    distances = distances.to_numpy()
    indices = indices.to_numpy()

    rho = rho[random_idx]

    min_dists = np.ones(n_points) * np.inf
    rho_argsort = np.argsort(rho)[::-1]
    X_sorted = X[rho_argsort]

    with mp.Pool(n_workers) as pool:
        tasks = [(i, X_sorted) for i in range(n_points)]
        min_dists = np.array(
            pool.starmap(
                min_dist_fn,
                tasks,
            )
        )

    min_dists[rho_argsort] = min_dists.copy()
    min_dists[np.isinf(min_dists)] = np.max(min_dists[~np.isinf(min_dists)])

    rho_cut = np.quantile(rho, 1 - np.sqrt(peak_pct))
    min_dist_cut = np.quantile(min_dists[rho > rho_cut], 1 - np.sqrt(peak_pct))

    peaks = np.argwhere((rho > rho_cut) & (min_dists > min_dist_cut)).flatten()

    def dist_to_nearest_peak_inner(x: np.ndarray) -> np.ndarray:
        dist_to_nearest_peak = np.ones(x.shape[0]) * np.inf
        dist_scales = distances[peaks].mean(axis=1)

        for peak_idx, peak in enumerate(peaks):
            dist_to_nearest_peak = np.minimum(
                dist_to_nearest_peak, dist_func(x, X[peak]) / dist_scales[peak_idx]
            )

        return dist_to_nearest_peak

    return dist_to_nearest_peak_inner
