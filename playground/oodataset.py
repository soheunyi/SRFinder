from __future__ import annotations

import pathlib
from typing import Iterable

import numpy as np
import pandas as pd


class SCDatasetInfo:
    def __init__(
        self,
        files: Iterable[pathlib.Path],
        inner_idxs: Iterable[Iterable[bool]],
        overwrite_features: dict[str, np.ndarray] = {},
    ):
        """
        SCDatasetInfo refers to "Simply concatenated dataset info".
        This class is used to store the information about the dataset that is split into multiple files.
        """
        assert len(files) == len(inner_idxs)
        assert all([file.exists() for file in files])
        dataset_length = sum([np.sum(inner_idx) for inner_idx in inner_idxs])
        assert all([len(v) == dataset_length for v in overwrite_features.values()])

        self.files = files
        self.inner_idxs = inner_idxs
        self.overwrite_features = overwrite_features

    def fetch_data(self):
        df_list = []
        for file_idx, file in enumerate(self.files):
            inner_idx = self.inner_idxs[file_idx]
            if np.sum(inner_idx) == 0:
                continue

            df = pd.read_hdf(file)
            assert len(df) == len(inner_idx)

            df_list.append(df[inner_idx])

        df = pd.concat(df_list).reset_index(drop=True)

        for k, v in self.overwrite_features.items():
            df[k] = v

        return df

    def __len__(self):
        return sum([np.sum(inner_idx) for inner_idx in self.inner_idxs])

    def __repr__(self):
        return f"SCDatasetInfo(files={self.files}, overwrite_features={self.overwrite_features.keys()}, length={len(self)})"
