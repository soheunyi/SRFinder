from typing import Literal
import warnings
import numpy as np
from tqdm import tqdm


def empirical_cdf(stats: np.ndarray, weights: np.ndarray):
    sorted_idx = np.argsort(stats)
    weights = weights[sorted_idx]
    stats = stats[sorted_idx]
    cdf = np.cumsum(weights)
    return stats, cdf / np.sum(weights)


def max_cdf_diff(
    stats_1: np.ndarray,
    stats_2: np.ndarray,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
):
    stats_merged = np.concatenate([stats_1, stats_2])
    weights_merged = np.concatenate([normalize(weights_1), normalize(weights_2)])
    labels_merged = np.concatenate([-np.ones_like(weights_1), np.ones_like(weights_2)])
    sorted_idx = np.argsort(stats_merged)
    weights_merged = weights_merged[sorted_idx]
    labels_merged = labels_merged[sorted_idx]

    cdf_diff = np.cumsum(weights_merged * labels_merged)
    max_diff = np.max(np.abs(cdf_diff))

    return max_diff


def mean_cdf_diff(
    stats_1: np.ndarray,
    stats_2: np.ndarray,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
):
    stats_merged = np.concatenate([stats_1, stats_2])
    weights_merged = np.concatenate([normalize(weights_1), normalize(weights_2)])
    labels_merged = np.concatenate([-np.ones_like(weights_1), np.ones_like(weights_2)])
    sorted_idx = np.argsort(stats_merged)
    weights_merged = weights_merged[sorted_idx]
    labels_merged = labels_merged[sorted_idx]

    cdf_diff = np.cumsum(weights_merged * labels_merged)
    mean_diff = np.mean(np.abs(cdf_diff))

    return mean_diff


def normalize(weights: np.ndarray):
    return weights / np.sum(weights)


def exponential_tilt(
    stats: np.ndarray,
    weights: np.ndarray,
    theta: float,
):
    tilt = weights * np.exp(theta * stats)
    return normalize(tilt)


def linear_tilt(stats: np.ndarray, weights: np.ndarray, theta: float):
    tilt = weights * (stats * theta + 1)
    return normalize(tilt)


def affine_tilt(stats: np.ndarray, weights: np.ndarray, slope: float, intercept: float):
    tilt = weights * (stats * slope + intercept)
    return normalize(tilt)


def max_cdf_diff_tilted(
    stats_1: np.ndarray,
    stats_2: np.ndarray,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
    theta: float,
    mode: str = "exponential",
):
    if mode == "linear":
        weights_2_tilted = np.copy(linear_tilt(stats_2, weights_2, theta))
    elif mode == "exponential":
        weights_2_tilted = np.copy(exponential_tilt(stats_2, weights_2, theta))
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return max_cdf_diff(stats_1, stats_2, weights_1, weights_2_tilted)


def min_sup_linear(
    slopes: np.ndarray,
    intercepts: np.ndarray,
    max_iter: int = 100,
    verbose: bool = False,
    atol: float = 1e-6,
):
    """
    Solves the following problem:
    min_t max_i (slopes[i] * t + intercepts[i])
    where t starts at t_start
    """
    assert len(slopes) == len(intercepts)
    assert slopes.ndim == 1
    assert intercepts.ndim == 1
    assert len(slopes) > 0

    # TODO: Return 0 if some NaN slopes or intercepts
    if np.isnan(slopes).any() or np.isnan(intercepts).any():
        warnings.warn("NaN slopes or intercepts")
        return 0

    # sort by slopes before starting
    sorted_idx = np.argsort(slopes)
    slopes = slopes[sorted_idx]
    intercepts = intercepts[sorted_idx]

    assert (
        slopes[0] * slopes[-1] < 0
    ), f"min and max of slopes must have different signs: {slopes[0]} * {slopes[-1]} = {slopes[0] * slopes[-1]}"
    assert slopes[0] < 0, f"min of slopes must be negative: {slopes[0]}"
    assert slopes[-1] > 0, f"max of slopes must be positive: {slopes[-1]}"

    i_left = 0
    i_right = len(slopes) - 1

    for counter in range(max_iter):
        t = (intercepts[i_right] - intercepts[i_left]) / (
            slopes[i_left] - slopes[i_right]
        )
        val_left = slopes[i_left] * t + intercepts[i_left]
        val_right = slopes[i_right] * t + intercepts[i_right]
        assert np.isclose(
            val_left, val_right, atol=atol
        ), "left and right must be equal"

        active_i = np.argmax(slopes * t + intercepts)
        if verbose:
            print(
                f"t = {t}, active_i = {active_i}, slopes[active_i] = {slopes[active_i]}, i_left = {i_left}, i_right = {i_right}"
            )

        if np.isclose(slopes[active_i] * t + intercepts[active_i], val_left, atol=atol):
            if verbose:
                print(f"Found minimum at t = {t}, iteration {counter}")
            break

        if slopes[active_i] > 0:
            i_right = active_i
        elif slopes[active_i] < 0:
            i_left = active_i
        else:
            break

        if i_right == i_left + 1:
            if verbose:
                print(f"Found minimum at t = {t}, iteration {counter}")
            break

        if verbose:
            print(
                f"t = {t}, i_left = {i_left}, i_right = {i_right}, slopes[i_left] = {slopes[i_left]}, slopes[i_right] = {slopes[i_right]}, intercepts[i_left] = {intercepts[i_left]}, intercepts[i_right] = {intercepts[i_right]}"
            )

    if counter == max_iter:
        raise Warning("Max number of iterations reached: ill-posed problem")

    return t


def emp_cdf_diff(
    stats_1: np.ndarray,
    stats_2: np.ndarray,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
):
    stats_all = np.concatenate([stats_1, stats_2])
    weights_all = np.concatenate([weights_1, weights_2])
    labels_all = np.concatenate([-np.ones_like(weights_1), np.ones_like(weights_2)])

    sorted_idx = np.argsort(stats_all)
    stats_all = stats_all[sorted_idx]
    weights_all = weights_all[sorted_idx]
    labels_all = labels_all[sorted_idx]

    cdf_diff = np.cumsum(weights_all * labels_all)
    cdf_diff_ = np.zeros_like(cdf_diff)
    cdf_diff_[sorted_idx] = cdf_diff

    return cdf_diff_[: len(stats_1)], cdf_diff_[len(stats_1) :]


def tilt_correction_via_first_order_approximation(
    stats_1: np.ndarray,
    stats_2: np.ndarray,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
    verbose: bool = False,
):
    """
    Solves the following problem:
    min_theta max_i |CDF_1(X_i) - CDF_2(X_i; theta)|
    where CDF_2(X_i; theta) approximates CDF of p_2(x) exp(theta * x)
    """

    weights_1 = normalize(weights_1)
    weights_2 = normalize(weights_2)

    stats_all = np.concatenate([stats_1, stats_2])
    weights_all = np.concatenate([weights_1, weights_2])
    labels_all = np.concatenate([-np.ones_like(weights_1), np.ones_like(weights_2)])

    sorted_idx = np.argsort(stats_all)
    stats_all = stats_all[sorted_idx]
    weights_all = weights_all[sorted_idx]
    labels_all = labels_all[sorted_idx]

    cdf_diff = np.cumsum(weights_all * labels_all)
    idx_2 = np.where(labels_all == 1)[0]

    intercepts = cdf_diff[idx_2]
    slopes = np.cumsum(weights_all[idx_2] * stats_all[idx_2]) - np.cumsum(
        weights_all[idx_2]
    ) * np.sum(weights_all[idx_2] * stats_all[idx_2])

    slopes = np.concatenate([slopes, -slopes])
    intercepts = np.concatenate([intercepts, -intercepts])

    theta = min_sup_linear(slopes, intercepts, verbose=verbose)

    return theta


def tilt_correction_iterative(
    stats_1: np.ndarray,
    stats_2: np.ndarray,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
    verbose: bool = False,
    max_iter: int = 100,
    d_theta_n_grid: int = 4,
):
    """
    Solves the following problem:
    min_theta max_i |CDF_1(X_i) - CDF_2(X_i; theta)|
    where CDF_2(X_i; theta) approximates CDF of p_2(x) exp(theta * x)
    """
    theta = 0

    weights_1 = normalize(weights_1)
    weights_2 = normalize(weights_2)

    for counter in range(max_iter):
        weights_2_it = exponential_tilt(stats_2, weights_2, theta)
        _, cdf_diff_2 = emp_cdf_diff(stats_1, stats_2, weights_1, weights_2_it)

        sorted_idx_2 = np.argsort(stats_2)
        weights_2_it_sorted = weights_2_it[sorted_idx_2]
        stats_2_sorted = stats_2[sorted_idx_2]

        slopes = np.cumsum(weights_2_it_sorted * stats_2_sorted) - np.cumsum(
            weights_2_it_sorted
        ) * np.sum(weights_2_it_sorted * stats_2_sorted)
        slopes = np.concatenate([slopes, -slopes])

        intercepts = cdf_diff_2[sorted_idx_2]
        intercepts = np.concatenate([intercepts, -intercepts])

        max_d_theta = min_sup_linear(slopes, intercepts, verbose=verbose)

        d_theta_grid = [
            max_d_theta * (i / d_theta_n_grid) for i in range(0, d_theta_n_grid + 1)
        ]
        d_theta_max_cdf_diff = [
            (
                dtheta,
                max_cdf_diff_tilted(
                    stats_1,
                    stats_2,
                    weights_1,
                    weights_2_it,
                    dtheta,
                    mode="exponential",
                ),
            )
            for dtheta in d_theta_grid
        ]
        min_idx = np.argmin([x[1] for x in d_theta_max_cdf_diff], axis=0)
        delta_theta = d_theta_grid[min_idx]
        max_diff_updated = d_theta_max_cdf_diff[min_idx][1]

        orig_max_diff = max_cdf_diff(stats_1, stats_2, weights_1, weights_2_it)

        if verbose:
            print(f"Iteration {counter}: {d_theta_max_cdf_diff}")

        if delta_theta != 0:
            if verbose:
                print(
                    f"Iteration {counter}: Updating theta: {theta} -> {theta + delta_theta}, max_diff: {orig_max_diff} -> {max_diff_updated}"
                )
            theta += delta_theta
        else:
            if verbose:
                print(
                    f"Iteration {counter}: No update, max_diff = {max_diff_updated}. Stopping."
                )
            break

    if counter == max_iter:
        raise Warning("Max number of iterations reached: ill-posed problem")

    return theta


def max_cdf_diff_permutation(
    stats_1: np.ndarray,
    stats_2: np.ndarray,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
    n_permutations: int = 1000,
    do_tqdm: bool = False,
    random_seed: int = 0,
    resample: bool = False,
    resample_ratio: float = 1.0,
):
    """
    Performs a permutation test for the max cdf diff.
    If resample is True, we use unit-weighted samples that are drawn from the empirical distribution of stats_1 and stats_2.
    """
    if resample:
        n_1 = int(resample_ratio * len(stats_1))
        n_2 = int(resample_ratio * len(stats_2))

        resample_idx_1 = np.random.choice(
            len(stats_1), size=n_1, replace=True, p=normalize(weights_1)
        )
        resample_idx_2 = np.random.choice(
            len(stats_2), size=n_2, replace=True, p=normalize(weights_2)
        )

        stats_1 = stats_1[resample_idx_1]
        stats_2 = stats_2[resample_idx_2]
        weights_1 = np.ones_like(stats_1) / n_1
        weights_2 = np.ones_like(stats_2) / n_2
    else:
        n_1 = len(stats_1)
        n_2 = len(stats_2)

    orig_max_diff = max_cdf_diff(stats_1, stats_2, weights_1, weights_2)

    stats_all = np.concatenate([stats_1, stats_2])
    weights_all = np.concatenate([weights_1, weights_2])

    max_cdf_diffs = []
    np.random.seed(random_seed)
    for _ in tqdm(range(n_permutations), disable=not do_tqdm):
        random_idx = np.random.permutation(len(stats_all))
        stats_1_perm = stats_all[random_idx[:n_1]]
        stats_2_perm = stats_all[random_idx[n_1:]]
        weights_1_perm = weights_all[random_idx[:n_1]]
        weights_2_perm = weights_all[random_idx[n_1:]]
        max_diff = max_cdf_diff(
            stats_1_perm, stats_2_perm, weights_1_perm, weights_2_perm
        )
        max_cdf_diffs.append(max_diff)

    p_value = np.sum(np.array(max_cdf_diffs) >= orig_max_diff) / n_permutations
    return orig_max_diff, max_cdf_diffs, p_value


def affine_correction(
    stats_3b: np.ndarray,
    stats_4b: np.ndarray,
    weights_3b: np.ndarray,
    weights_4b: np.ndarray,
    grid_size: float = 0.001,
    cdf_mode: Literal["max", "mean"] = "max",
):

    # 1. Center stats_3b
    mean_3b = np.sum(weights_3b * stats_3b) / np.sum(weights_3b)
    std_3b = np.sqrt(
        np.sum(weights_3b * (stats_3b - mean_3b) ** 2) / np.sum(weights_3b)
    )
    stats_3b_centered = (stats_3b - mean_3b) / std_3b

    # 2. Set correction_min and correction_max
    correction_slope_max = -1 / np.min(stats_3b_centered)
    correction_slope_min = -1 / np.max(stats_3b_centered)

    print(
        f"correction_slope_min: {correction_slope_min}, correction_slope_max: {correction_slope_max}"
    )

    # 3. Find the correction that minimizes the max cdf diff
    correction_grid = [
        grid_size * k
        for k in range(
            int(correction_slope_min / grid_size),
            int(correction_slope_max / grid_size) + 1,
        )
    ]
    cdf_diff_grid = np.zeros(len(correction_grid))
    for i, correction_slope in enumerate(correction_grid):
        if cdf_mode == "max":
            cdf_diff_grid[i] = max_cdf_diff(
                stats_3b,
                stats_4b,
                affine_tilt(stats_3b_centered, weights_3b, correction_slope, 1),
                weights_4b,
            )
        elif cdf_mode == "mean":
            cdf_diff_grid[i] = mean_cdf_diff(
                stats_3b,
                stats_4b,
                affine_tilt(stats_3b_centered, weights_3b, correction_slope, 1),
                weights_4b,
            )
        else:
            raise ValueError(f"Invalid cdf_mode: {cdf_mode}")
    min_idx = np.argmin(cdf_diff_grid)
    correction_slope = correction_grid[min_idx]

    final_slope = correction_slope / std_3b
    final_intercept = 1 - final_slope * mean_3b

    return final_slope, final_intercept


def p_value_ks(
    stats_1: np.ndarray,
    stats_2: np.ndarray,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
    norm_mode: Literal["N", "W", "W^2", "(W)^2 / W^2"] = "(W)^2 / W^2",
):

    if norm_mode == "N":
        N1 = len(stats_1)
        N2 = len(stats_2)
    elif norm_mode == "W":
        N1 = np.sum(weights_1)
        N2 = np.sum(weights_2)
    elif norm_mode == "W^2":
        N1 = np.sum(weights_1**2)
        N2 = np.sum(weights_2**2)
    elif norm_mode == "(W)^2 / W^2":
        N1 = np.sum(weights_1) ** 2 / np.sum(weights_1**2)
        N2 = np.sum(weights_2) ** 2 / np.sum(weights_2**2)
    else:
        raise ValueError(f"Invalid norm_mode: {norm_mode}")

    max_diff = max_cdf_diff(stats_1, stats_2, weights_1, weights_2)
    norm = np.sqrt((N1 + N2) / (N1 * N2))
    return min(1.0, 2 * np.exp(-2 * (max_diff / norm) ** 2))
