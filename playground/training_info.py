from __future__ import annotations
import pathlib
import random
import string
import pickle
from typing import Iterable

import numpy as np
import pandas as pd

from dataset import DatasetInfo

# get current directory of the file
SAVE_DIR = pathlib.Path(__file__).parent / "data/training_info"
# check if SAVE_DIR exists, if not, create it
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def create_hash(directory: pathlib.Path) -> str:
    # create a new hash that is not already in the directory
    files = directory.glob("*")
    hashes = [file.name for file in files]
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=12))
    while random_string in hashes:
        random_string = "".join(
            random.choices(string.ascii_letters + string.digits, k=12)
        )
    return random_string


class TrainingInfo:
    def __init__(self, hparams: dict, dinfo_train: DatasetInfo, dinfo_val: DatasetInfo):
        self._hparams = hparams
        self.dinfo_train = dinfo_train
        self.dinfo_val = dinfo_val
        self._hash = create_hash(SAVE_DIR)
        self._aux_info = {}

    @property
    def hparams(self):
        return self._hparams

    @property
    def hash(self):
        return self._hash

    @property
    def aux_info(self):
        return self._aux_info

    def update_aux_info(self, key: str, value):
        self._aux_info[key] = value

    def save(self):
        with open(SAVE_DIR / self.hash, "wb") as f:
            pickle.dump(self, f)

    def get_training_data(self):
        return self.dinfo_train.fetch_data(), self.dinfo_val.fetch_data()

    @classmethod
    def load(cls, hash: str) -> TrainingInfo:
        with open(SAVE_DIR / hash, "rb") as f:
            return pickle.load(f)

    @classmethod
    def get_existing_hparams(cls):
        hparams_list = []
        for file in SAVE_DIR.glob("*"):
            tinfo = TrainingInfo.load(file)
            hparams_list.append(tinfo.hparams)

        return hparams_list

    @classmethod
    def find(cls, hparam_filter: dict):
        hashes = []
        for file in SAVE_DIR.glob("*"):
            tinfo = TrainingInfo.load(file)
            is_match = True
            for key, value in hparam_filter.items():
                if tinfo.hparams.get(key) != value:
                    is_match = False
                    break
            if is_match:
                hashes.append(tinfo.hash)

        return hashes
