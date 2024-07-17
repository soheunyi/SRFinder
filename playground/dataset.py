from __future__ import annotations

import pathlib
from typing import Iterable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def split_dataset(
    X: torch.Tensor,
    y: torch.Tensor,
    valid_split: float,
    train_batch_size: int = 32,
    valid_batch_size: int = 32,
    seed: int = 42,
):
    np.random.seed(seed)
    n_samples = X.shape[0]
    n_valid = int(n_samples * valid_split)
    valid_indices = np.random.choice(n_samples, n_valid, replace=False)

    X_train, y_train = X[~valid_indices], y[~valid_indices]
    X_valid, y_valid = X[valid_indices], y[valid_indices]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=train_batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        TensorDataset(X_valid, y_valid),
        batch_size=valid_batch_size,
        shuffle=True,
    )

    return train_loader, valid_loader


class DatasetInfo:
    def __init__(
        self,
        files: Iterable[pathlib.Path],
        file_idx: np.ndarray,
        inner_idx: np.ndarray,
        overwrite_features: dict[str, np.ndarray] = {},
    ):
        assert len(file_idx) == len(inner_idx)
        assert max(file_idx) + 1 == len(files)
        # check if the file exists
        assert all([file.exists() for file in files])
        assert all([len(v) == len(inner_idx) for v in overwrite_features.values()])

        self.files = files
        self.file_idx = file_idx
        self.inner_idx = inner_idx
        self.overwrite_features = overwrite_features

    def fetch_data(self):
        sort_idx = np.arange(len(self.inner_idx))

        df_list = []
        for fi in np.unique(self.file_idx):
            filename = self.files[fi]
            df = pd.read_hdf(filename)
            assert "sort_idx_" not in df.columns

            idx = self.file_idx == fi
            inner_idx = self.inner_idx[idx]
            df = df.iloc[inner_idx]
            df["sort_idx_"] = sort_idx[idx]
            df_list.append(df)

        df = (
            pd.concat(df_list)
            .sort_values("sort_idx_")
            .drop(columns="sort_idx_")
            .reset_index(drop=True)
        )
        for k, v in self.overwrite_features.items():
            df[k] = v

        return df

    def __len__(self):
        return len(self.inner_idx)

    def get(self, idx: Iterable[int]) -> DatasetInfo:
        return DatasetInfo(
            self.files,
            self.file_idx[idx],
            self.inner_idx[idx],
            {k: v[idx] for k, v in self.overwrite_features.items()},
        )

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.get(range(*idx.indices(len(self))))
        elif isinstance(idx, Iterable):
            return self.get(idx)
        else:
            return self.get([idx])

    def __repr__(self):
        return f"DatasetInfo(files={self.files},overwrite_features={self.overwrite_features.keys()}, length={len(self)})"


def generate_tt_dataset(
    seed: int, n_3b: int, n_all4b: int, signal_ratio: float, test_ratio: float
):
    directory = pathlib.Path("../events/MG3")
    np.random.seed(seed)

    file_idxs = []
    inner_idxs = []

    n_signal = int(n_all4b * signal_ratio)
    n_bg4b = n_all4b - n_signal

    df_3b = pd.read_hdf(directory / "dataframes" / "threeTag_picoAOD.h5")
    assert n_3b <= len(df_3b)
    indices_3b = np.random.choice(len(df_3b), n_3b, replace=False)
    df_3b = df_3b.iloc[indices_3b]
    inner_idxs.append(indices_3b)
    file_idxs.append(np.zeros(len(df_3b), dtype=int))

    df_bg4b = pd.read_hdf(directory / "dataframes" / "fourTag_10x_picoAOD.h5")
    assert n_bg4b <= len(df_bg4b)
    indices_bg4b = np.random.choice(len(df_bg4b), n_bg4b, replace=False)
    df_bg4b = df_bg4b.iloc[indices_bg4b]
    inner_idxs.append(indices_bg4b)
    file_idxs.append(np.ones(len(df_bg4b), dtype=int))

    df_signal = pd.read_hdf(directory / "dataframes" / "HH4b_picoAOD.h5")
    assert n_signal <= len(df_signal)
    indices_signal = np.random.choice(len(df_signal), n_signal, replace=False)
    df_signal = df_signal.iloc[indices_signal]
    inner_idxs.append(indices_signal)
    file_idxs.append(2 * np.ones(len(df_signal), dtype=int))

    df_3b["signal"] = False
    df_bg4b["signal"] = False
    df_signal["signal"] = True

    # reweighting in order to fit signal_ratio
    total_w_bg4b = df_bg4b["weight"].sum()
    total_w_signal = df_signal["weight"].sum()

    df_signal["weight"] *= (signal_ratio / (1 - signal_ratio)) * (
        total_w_bg4b / total_w_signal
    )

    df = pd.concat([df_3b, df_bg4b, df_signal])
    files = [
        directory / "dataframes" / "threeTag_picoAOD.h5",
        directory / "dataframes" / "fourTag_10x_picoAOD.h5",
        directory / "dataframes" / "HH4b_picoAOD.h5",
    ]
    file_idx = np.concatenate(file_idxs)
    inner_idx = np.concatenate(inner_idxs)
    overwrite_features = {"signal": df["signal"].values, "weight": df["weight"].values}

    # includes all test and train data
    dinfo = DatasetInfo(files, file_idx, inner_idx, overwrite_features)
    # shuffle dinfo
    dinfo = dinfo[np.random.permutation(len(dinfo))]
    n_test = int(test_ratio * len(dinfo))
    dinfo_train = dinfo[n_test:]
    dinfo_test = dinfo[:n_test]

    return dinfo_train, dinfo_test
