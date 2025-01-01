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
    sorted_indices = np.argsort(x_values)
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
