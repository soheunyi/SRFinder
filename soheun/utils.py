from datetime import datetime
import pathlib
import random
import string
import time

import numpy as np


def require_keys(config: dict, keys: list):
    for key in keys:
        if key not in config:
            raise ValueError(f"Key {key} is missing in the config")


def create_hash(directory: pathlib.Path) -> str:
    # create a new hash that is not already in the directory
    # get current timestamp
    files = directory.glob("*")
    existing_hashes = [file.name for file in files]

    def create_hash_with_timestamp():
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        random.seed(timestamp)
        random_string = "".join(
            random.choices(string.ascii_letters + string.digits, k=6)
        )
        return timestamp + "_" + random_string

    hash_ = create_hash_with_timestamp()

    while hash_ in existing_hashes:
        hash_ = create_hash_with_timestamp()

    return hash_


def get_quantiles_with_weights(
    x_values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    assert len(x_values) == len(weights)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1)
    assert np.all(weights >= 0)
    assert np.sum(weights) > 0

    # normalize weights
    weights = weights / np.sum(weights)
    sorted_indices = np.argsort(x_values, kind="stable")
    sorted_x_values = x_values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    cumsum_weights = np.cumsum(sorted_weights)
    return np.interp(quantiles, cumsum_weights, sorted_x_values)


def safe_dict(d: dict | None, key: str, default=None):
    if isinstance(d, dict):
        if key not in d:
            d[key] = default
        return d[key]
    else:
        return default


def select_random_true_elements(
    idx: np.ndarray[bool], ratio: float, seed: int
) -> np.ndarray[bool]:
    true_idx_int = np.where(idx)[0]
    np.random.seed(seed)
    np.random.shuffle(true_idx_int)
    selected_idx_int = true_idx_int[: int(len(true_idx_int) * ratio)]
    selected_idx = np.zeros_like(idx, dtype=bool)
    selected_idx[selected_idx_int] = True
    return selected_idx


def test_select_random_true_elements():
    ratio = 0.5
    n_true = 100
    n_false = 100
    n_expected_true = int(n_true * ratio)

    for seed in range(10):
        idx = np.array([True] * n_true + [False] * n_false)
        np.random.seed(seed)
        np.random.shuffle(idx)
        selected_idx = select_random_true_elements(idx, ratio, seed)

        assert (
            np.sum(selected_idx) == n_expected_true
        ), f"The number of selected elements is incorrect for seed {seed}"
        assert np.all(
            idx[selected_idx] == True
        ), f"Selected elements should be a subset of the original True elements for seed {seed}"
        assert np.array_equal(
            select_random_true_elements(idx, ratio, seed),
            select_random_true_elements(idx, ratio, seed),
        ), f"The function should be deterministic given the same seed for seed {seed}"


if __name__ == "__main__":
    test_select_random_true_elements()
