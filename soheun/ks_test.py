import numpy as np


def max_cdf_diff(
    stats_1: np.ndarray,
    stats_2: np.ndarray,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
):
    stats_merged = np.concatenate([stats_1, stats_2])
    weights_merged = np.concatenate([normalize(weights_1), normalize(weights_2)])
    labels_merged = np.concatenate([np.ones_like(weights_1), -np.ones_like(weights_2)])
    sorted_idx = np.argsort(stats_merged)
    weights_merged = weights_merged[sorted_idx]
    labels_merged = labels_merged[sorted_idx]

    cdf_diff = np.cumsum(weights_merged * labels_merged)
    max_diff = np.max(np.abs(cdf_diff))

    return max_diff


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


def max_cdf_diff_tilted(
    stats_1: np.ndarray,
    stats_2: np.ndarray,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
    theta: float,
    mode: str = "exponential",
):
    if mode == "linear":
        weights_2 = linear_tilt(stats_2, weights_2, theta)
    elif mode == "exponential":
        weights_2 = exponential_tilt(stats_2, weights_2, theta)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return max_cdf_diff(stats_1, stats_2, weights_1, weights_2)


def min_sup_linear(
    slopes: np.ndarray,
    intercepts: np.ndarray,
    max_iter: int = 100,
    verbose: bool = False,
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

    # sort by slopes before starting
    sorted_idx = np.argsort(slopes)
    slopes = slopes[sorted_idx]
    intercepts = intercepts[sorted_idx]

    assert slopes[0] * slopes[-1] < 0, "min and max of slopes must have different signs"
    assert slopes[0] < 0, "min of slopes must be negative"
    assert slopes[-1] > 0, "max of slopes must be positive"

    i_left = 0
    i_right = len(slopes) - 1

    for counter in range(max_iter):
        t = (intercepts[i_right] - intercepts[i_left]) / (
            slopes[i_left] - slopes[i_right]
        )
        active_i = np.argmax(slopes * t + intercepts)
        if verbose:
            print(
                f"t = {t}, active_i = {active_i}, slopes[active_i] = {slopes[active_i]}, i_left = {i_left}, i_right = {i_right}"
            )

        val_left = slopes[i_left] * t + intercepts[i_left]
        val_right = slopes[i_right] * t + intercepts[i_right]
        assert np.isclose(val_left, val_right), "left and right must be equal"

        if np.isclose(
            slopes[active_i] * t + intercepts[active_i],
            val_left,
        ):
            print(f"Found minimum at t = {t}, iteration {counter}")
            break

        if slopes[active_i] > 0:
            i_right = active_i
        elif slopes[active_i] < 0:
            i_left = active_i
        else:
            break

        if i_right == i_left + 1:
            print(f"Found minimum at t = {t}, iteration {counter}")
            break

        if verbose:
            print(
                f"t = {t}, i_left = {i_left}, i_right = {i_right}, slopes[i_left] = {slopes[i_left]}, slopes[i_right] = {slopes[i_right]}, intercepts[i_left] = {intercepts[i_left]}, intercepts[i_right] = {intercepts[i_right]}"
            )

    if counter == max_iter:
        raise Warning("Max number of iterations reached: ill-posed problem")

    return t


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
