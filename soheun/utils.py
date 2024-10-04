from datetime import datetime
import pathlib
import random
import string
import time


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
