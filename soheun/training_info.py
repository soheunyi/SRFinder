from __future__ import annotations
from copy import deepcopy
import pathlib
import pickle
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import tqdm
from sklearn.model_selection import train_test_split
import re

from dataset import SCDatasetInfo, split_scdinfo_multiple, MotherSamples
from utils import create_hash, require_keys

from fvt_classifier import FvTClassifier
from attention_classifier import AttentionClassifier
from smearing import smear_features

TINFO_SAVE_DIR = pathlib.Path(__file__).parent / "data/TrainingInfo"
TINFO_SAVE_DIR.mkdir(parents=True, exist_ok=True)
TINFO_META_DIR = pathlib.Path(__file__).parent / "data/metadata/TrainingInfo.pkl"


class TrainingInfo:
    SAVE_DIR = TINFO_SAVE_DIR
    META_DIR = TINFO_META_DIR

    def __init__(
        self,
        hparams: dict,
        ms_hash: str,
        ms_idx: np.ndarray[bool],
    ):
        """
        Save the hparams, scdinfo, and seed for training models.
        hparams must include val_ratio, data_seed (for reproducing train/val data),
        """
        self._hparams = hparams
        # raise value error if any key of hparams starts with "aux_info"
        for key in hparams.keys():
            if re.match(r"^aux_info.*", key):
                raise ValueError(f"Key {key} starts with 'aux_info'")
        assert "data_seed" in hparams
        assert "val_ratio" in hparams
        assert "model" in hparams

        self.data_seed = hparams["data_seed"]
        self.val_ratio = hparams["val_ratio"]
        self.model = hparams["model"]

        self._ms_hash = ms_hash
        self._ms_idx = ms_idx

        self._hash = create_hash(TrainingInfo.SAVE_DIR)
        self._aux_info = {}

    def load_trained_model(self, mode: str) -> torch.nn.Module:
        assert self.model in ["FvTClassifier", "AttentionClassifier"]
        assert mode in ["best", "last"]

        ckpt_dir = pathlib.Path(__file__).parent / "data/checkpoints"
        name = f"{self._hash}_{mode}.ckpt"

        if self.model == "FvTClassifier":
            return FvTClassifier.load_from_checkpoint(ckpt_dir / name)
        elif self.model == "AttentionClassifier":
            return AttentionClassifier.load_from_checkpoint(ckpt_dir / name)
        else:
            raise ValueError(f"Model {self.model} not found")

    @property
    def aux_info(self) -> dict:
        return self._aux_info

    @property
    def base_fvt_model(self) -> FvTClassifier:
        require_keys(self.hparams, ["encoder_hash", "encoder_mode"])
        encoder_tinfo = TrainingInfo.load(self.hparams["encoder_hash"])
        base_fvt_model = encoder_tinfo.load_trained_model(self.hparams["encoder_mode"])
        base_fvt_model.eval()
        return base_fvt_model

    @property
    def ms_hash(self) -> str:
        return self._ms_hash

    @property
    def ms_idx(self) -> np.ndarray[bool]:
        return self._ms_idx

    @property
    def mother_samples(self) -> MotherSamples:
        return MotherSamples.load(self._ms_hash)

    @property
    def scdinfo(self) -> SCDatasetInfo:
        mother_samples = MotherSamples.load(self._ms_hash)
        return mother_samples.scdinfo[self._ms_idx]

    def fetch_train_val_smeared_features(
        self,
        features: Iterable[str],
        label: str,
        weight: str,
        label_dtype: str = torch.long,
        device=torch.device("cuda"),
    ) -> tuple[TensorDataset, TensorDataset]:
        require_keys(self.hparams, ["smearing", "encoder_hash", "encoder_mode"])
        require_keys(
            self.hparams["smearing"],
            ["noise_scale", "seed", "hard_cutoff", "scale_mode"],
        )

        base_fvt_model = self.base_fvt_model
        base_fvt_model.to(device)

        df_all = self.scdinfo.fetch_data()
        df_all = df_all.sample(frac=1, random_state=self.data_seed).reset_index(
            drop=True
        )
        X_torch = torch.tensor(df_all[features].values, dtype=torch.float32)
        q_repr = base_fvt_model.representations(X_torch)[0]
        q_repr_smear = smear_features(
            q_repr,
            noise_scale=self.hparams["smearing"]["noise_scale"],
            seed=self.hparams["smearing"]["seed"],
            hard_cutoff=self.hparams["smearing"]["hard_cutoff"],
            scale_mode=self.hparams["smearing"]["scale_mode"],
        )

        (q_repr_smear_train, q_repr_smear_val, y_train, y_val, w_train, w_val) = (
            train_test_split(
                q_repr_smear,
                df_all[label].values,
                df_all[weight].values,
                test_size=self.val_ratio,
                random_state=self.data_seed,
            )
        )
        if type(q_repr_smear_train) == np.ndarray:
            q_repr_smear_train = torch.tensor(q_repr_smear_train, dtype=torch.float32)
        else:
            assert type(q_repr_smear_train) == torch.Tensor
        if type(q_repr_smear_val) == np.ndarray:
            q_repr_smear_val = torch.tensor(q_repr_smear_val, dtype=torch.float32)
        else:
            assert type(q_repr_smear_val) == torch.Tensor

        y_train = torch.tensor(y_train, dtype=label_dtype)
        y_val = torch.tensor(y_val, dtype=label_dtype)
        w_train = torch.tensor(w_train, dtype=torch.float32)
        w_val = torch.tensor(w_val, dtype=torch.float32)

        fit_batch_size = self.hparams.get("fit_batch_size", 0)
        if fit_batch_size > 0:
            q_repr_smear_train = q_repr_smear_train[
                : (len(q_repr_smear_train) // fit_batch_size) * fit_batch_size
            ]
            q_repr_smear_val = q_repr_smear_val[
                : (len(q_repr_smear_val) // fit_batch_size) * fit_batch_size
            ]
            y_train = y_train[: (len(y_train) // fit_batch_size) * fit_batch_size]
            y_val = y_val[: (len(y_val) // fit_batch_size) * fit_batch_size]
            w_train = w_train[: (len(w_train) // fit_batch_size) * fit_batch_size]
            w_val = w_val[: (len(w_val) // fit_batch_size) * fit_batch_size]

        return (
            TensorDataset(q_repr_smear_train, y_train, w_train),
            TensorDataset(q_repr_smear_val, y_val, w_val),
        )

    def fetch_train_val_scdinfo(self) -> tuple[SCDatasetInfo, SCDatasetInfo]:
        """
        Return the training and validation scdinfo.
        """

        scdinfo_train, scdinfo_val = split_scdinfo_multiple(
            self.scdinfo, [1 - self.val_ratio, self.val_ratio], self.data_seed
        )
        return scdinfo_train, scdinfo_val

    def fetch_train_val_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return the training and validation data.
        """
        scdinfo_train, scdinfo_val = self.fetch_train_val_scdinfo()

        df_train, df_val = SCDatasetInfo.fetch_multiple_data(
            [scdinfo_train, scdinfo_val]
        )

        df_train = df_train.sample(frac=1, random_state=self.data_seed).reset_index(
            drop=True
        )
        df_val = df_val.sample(frac=1, random_state=self.data_seed).reset_index(
            drop=True
        )

        fit_batch_size = self.hparams.get("fit_batch_size", 0)
        if fit_batch_size > 0:
            df_train = df_train.iloc[
                : (len(df_train) // fit_batch_size) * fit_batch_size
            ]
            df_val = df_val.iloc[: (len(df_val) // fit_batch_size) * fit_batch_size]

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
        hparams = deepcopy(self._hparams)
        for key in self._aux_info.keys():
            hparams["_".join(["aux_info", key])] = self._aux_info[key]
        return hparams

    @property
    def hash(self):
        return self._hash

    @property
    def aux_info(self):
        return self._aux_info

    def update_aux_info(self, **kwargs):
        self._aux_info.update(kwargs)

    def save(self):
        print(f"Saving Training Info: {self.hash}")
        with open(TrainingInfo.SAVE_DIR / self.hash, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(hash: str) -> TrainingInfo:
        with open(TrainingInfo.SAVE_DIR / hash, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def get_existing_hparams():
        _, hparams = TrainingInfo.find({}, return_hparams=True)
        return hparams

    @staticmethod
    def load_metadata() -> dict[str, dict]:
        with open(TrainingInfo.META_DIR, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def update_metadata():
        with open(TrainingInfo.META_DIR, "wb") as f:
            hashes, hparams = TrainingInfo.find(
                {}, return_hparams=True, from_metadata=False
            )
            pickle.dump(dict(zip(hashes, hparams)), f)

    @staticmethod
    def find(
        hparam_filter: dict[str, any],
        return_hparams=False,
        sort_by: list[str] = [],
        from_metadata=True,
    ) -> list[str] | tuple[list[str], list[dict[str, any]]]:

        def is_match(hparam: dict[str, any]) -> bool:
            for key, value in hparam_filter.items():
                if callable(value):
                    if not value(hparam.get(key)):
                        return False
                elif hparam.get(key) != value:
                    return False
            return True

        hash_hparams = []
        if from_metadata:
            all_hash_hparams = TrainingInfo.load_metadata()
            for hash_, hparams in all_hash_hparams.items():
                if is_match(hparams):
                    hash_hparams.append((hash_, hparams))
        else:
            for file in tqdm.tqdm(TrainingInfo.SAVE_DIR.glob("*")):
                tinfo = TrainingInfo.load(file)
                if is_match(tinfo.hparams):
                    hash_hparams.append((tinfo.hash, tinfo.hparams))

        if len(hash_hparams) == 0:
            hashes, hparams = [], []
        else:
            hashes, hparams = zip(*hash_hparams)

        # sort by the given keys
        if len(sort_by) > 0 and len(hashes) > 0:
            for hp in hparams:
                assert all(
                    [key in hp for key in sort_by]
                ), f"sort_by keys {sort_by} not in hparams {hp}"

            argsort = np.lexsort(tuple([hp[key] for hp in hparams] for key in sort_by))
            hashes = [hashes[i] for i in argsort]
            hparams = [hparams[i] for i in argsort]

        if return_hparams:
            return hashes, hparams
        else:
            return hashes
