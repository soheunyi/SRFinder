from __future__ import annotations
import pathlib
import random
import string
import pickle
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm

from dataset import DatasetInfo, SCDatasetInfo, split_scdinfo

# get current directory of the file
TINFO_SAVE_DIR = pathlib.Path(__file__).parent / "data/training_info"
# check if TINFO_SAVE_DIR exists, if not, create it
TINFO_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def create_hash(directory: pathlib.Path) -> str:
    # create a new hash that is not already in the directory
    files = directory.glob("*")
    hashes = [file.name for file in files]
    random_string = "".join(random.choices(
        string.ascii_letters + string.digits, k=12))
    while random_string in hashes:
        random_string = "".join(
            random.choices(string.ascii_letters + string.digits, k=12)
        )
    return random_string


class TrainingInfo:
    SAVE_DIR = TINFO_SAVE_DIR

    def __init__(self, hparams: dict, dinfo_train: DatasetInfo, dinfo_val: DatasetInfo):
        self._hparams = hparams
        self.dinfo_train = dinfo_train
        self.dinfo_val = dinfo_val
        self._hash = create_hash(TINFO_SAVE_DIR)
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
        with open(TINFO_SAVE_DIR / self.hash, "wb") as f:
            pickle.dump(self, f)

    def get_training_data(self):
        return self.dinfo_train.fetch_data(), self.dinfo_val.fetch_data()

    @staticmethod
    def load(hash: str) -> TrainingInfo:
        with open(TINFO_SAVE_DIR / hash, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def get_existing_hparams():
        hparams_list = []
        for file in TINFO_SAVE_DIR.glob("*"):
            tinfo = TrainingInfo.load(file)
            hparams_list.append(tinfo.hparams)

        return hparams_list

    @staticmethod
    def find(hparam_filter: dict, return_hparams=False):
        hashes = []
        for file in tqdm.tqdm(TINFO_SAVE_DIR.glob("*")):
            tinfo = TrainingInfo.load(file)
            is_match = True
            for key, value in hparam_filter.items():
                if callable(value):
                    if not value(tinfo.hparams.get(key)):
                        is_match = False
                        break
                elif tinfo.hparams.get(key) != value:
                    is_match = False
                    break
            if is_match:
                if return_hparams:
                    hashes.append((tinfo.hash, tinfo.hparams))
                else:
                    hashes.append(tinfo.hash)

        if return_hparams:
            hashes, hparams = zip(*hashes)
            return hashes, hparams
        else:
            return hashes


class TrainingInfoV2:
    SAVE_DIR = TINFO_SAVE_DIR

    def __init__(self, hparams: dict, scdinfo: SCDatasetInfo):
        """
        Save the hparams, scdinfo, and seed for training models.
        hparams must include val_ratio, data_seed (for reproducing train/val data),
        """
        self._hparams = hparams
        assert "data_seed" in hparams
        assert "val_ratio" in hparams
        self.data_seed = hparams["data_seed"]
        self.val_ratio = hparams["val_ratio"]
        self.scdinfo = scdinfo
        self._hash = create_hash(TINFO_SAVE_DIR)
        self._aux_info = {}

    def fetch_train_val_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return the training and validation data.
        """
        scdinfo_train, scdinfo_val = split_scdinfo(
            self.scdinfo, 1 - self.val_ratio, self.data_seed
        )

        df_train, df_val = SCDatasetInfo.fetch_multiple_data(
            [scdinfo_train, scdinfo_val]
        )

        df_train = df_train.sample(frac=1, random_state=self.data_seed).reset_index(
            drop=True
        )
        df_val = df_val.sample(frac=1, random_state=self.data_seed).reset_index(
            drop=True
        )

        fit_batch_size = self.hparams.get("fit_batch_size", False)
        if fit_batch_size:
            assert (
                "batch_size" in self.hparams
            ), "batch_size must be in hparams if fit_batch_size is True"
            batch_size = self.hparams["batch_size"]
            df_train = df_train.iloc[: (
                len(df_train) // batch_size) * batch_size]
            df_val = df_val.iloc[: (len(df_val) // batch_size) * batch_size]

        return df_train, df_val

    def fetch_train_val_tensor_datasets(
        self,
        features: Iterable[str],
        label: str,
        weight: str,
        label_dtype: str = torch.long,
    ) -> tuple[TensorDataset, TensorDataset]:
        """
        fetch train and val dataloader
        """
        df_train, df_val = self.fetch_train_val_data()

        X_train = torch.tensor(df_train[features].values, dtype=torch.float32)
        y_train = torch.tensor(df_train[label].values, dtype=label_dtype)
        w_train = torch.tensor(df_train[weight].values, dtype=torch.float32)

        X_val = torch.tensor(df_val[features].values, dtype=torch.float32)
        y_val = torch.tensor(df_val[label].values, dtype=label_dtype)
        w_val = torch.tensor(df_val[weight].values, dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train, w_train)
        val_dataset = TensorDataset(X_val, y_val, w_val)

        return train_dataset, val_dataset

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
        print(f"Saving Training Info: {self.hash}")
        with open(TINFO_SAVE_DIR / self.hash, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(hash: str) -> TrainingInfoV2:
        with open(TINFO_SAVE_DIR / hash, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def get_existing_hparams():
        hparams_list = []
        for file in TINFO_SAVE_DIR.glob("*"):
            tinfo = TrainingInfoV2.load(file)
            hparams_list.append(tinfo.hparams)

        return hparams_list

    @staticmethod
    def find(hparam_filter: dict):
        hashes = []
        for file in TINFO_SAVE_DIR.glob("*"):
            tinfo = TrainingInfoV2.load(file)
            is_match = True
            for key, value in hparam_filter.items():
                if tinfo.hparams.get(key) != value:
                    is_match = False
                    break
            if is_match:
                hashes.append(tinfo.hash)

        return hashes
