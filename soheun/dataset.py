from __future__ import annotations

import pathlib
import pickle
from typing import Iterable
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import tqdm

from utils import require_keys, create_hash


class DatasetInfo:
    def __init__(
        self,
        files: Iterable[pathlib.Path],
        file_idx: np.ndarray,
        inner_idx: np.ndarray,
        overwrite_features: dict[str, np.ndarray] = {},
    ):
        assert len(file_idx) == len(inner_idx)
        assert max(file_idx) + 1 <= len(files)
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
            if np.sum(idx) == 0:
                continue
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
        return f"DatasetInfo(files={self.files}, overwrite_features={self.overwrite_features.keys()}, length={len(self)})"


class SCDatasetInfo:
    def __init__(
        self,
        files: Iterable[pathlib.Path],
        inner_idxs: Iterable[Iterable[bool]],
    ):
        """
        SCDatasetInfo refers to "Simply concatenated dataset info".
        This class is used to store the information about the dataset that is split into multiple files.
        Better in terms of memory usage than DatasetInfo.
        Does not allow overwritting features that are not present in the files due to memory concerns.
        """
        assert len(files) == len(inner_idxs)
        assert all([file.exists() for file in files])

        self.files = files
        self.inner_idxs = inner_idxs

    def compare(self, other: SCDatasetInfo) -> bool:
        return all(
            [
                self.files == other.files,
                all(
                    [
                        np.all(self.inner_idxs[i] == other.inner_idxs[i])
                        for i in range(len(self.files))
                    ]
                ),
            ]
        )

    @staticmethod
    def fetch_multiple_data(scdinfos: Iterable[SCDatasetInfo]) -> list[pd.DataFrame]:
        """
        To reduce loading time, fetches data from multiple SCDatasetInfo objects.
        """
        assert len(scdinfos) > 0
        assert all(
            [len(scdinfo.files) == len(scdinfos[0].files) for scdinfo in scdinfos]
        )
        assert all(
            [
                all(
                    [
                        scdinfos[0].files[j] == scdinfo.files[j]
                        for j in range(len(scdinfo.files))
                    ]
                )
                for scdinfo in scdinfos
            ]
        )

        original_df_list = [pd.read_hdf(file) for file in scdinfos[0].files]
        df_list = []

        for scdinfo in scdinfos:
            df_list_inner = []
            for fi, inner_idx in enumerate(scdinfo.inner_idxs):
                if np.sum(inner_idx) == 0:
                    continue

                df_inner = original_df_list[fi]
                df_list_inner.append(df_inner[inner_idx])

            df = pd.concat(df_list_inner).reset_index(drop=True)
            df_list.append(df)

        return df_list

    def fetch_data(self):
        df_list = []
        for file, inner_idx in zip(self.files, self.inner_idxs):
            if np.sum(inner_idx) == 0:
                continue

            df = pd.read_hdf(file)
            assert len(df) == len(inner_idx)

            df_list.append(df[inner_idx])

        df = pd.concat(df_list).reset_index(drop=True)

        return df

    def to_dataset_info(
        self, overwrite_features: dict[str, np.ndarray] = {}
    ) -> DatasetInfo:
        file_idx = np.concatenate(
            [
                np.full(np.sum(inner_idx), fi)
                for fi, inner_idx in enumerate(self.inner_idxs)
            ]
        )
        inner_idx = np.concatenate(
            [np.where(self.inner_idxs[fi])[0] for fi in range(len(self.files))]
        )
        # assert all([len(v) == len(inner_idx) for v in overwrite_features.values()])

        return DatasetInfo(self.files, file_idx, inner_idx, overwrite_features)

    def check_validity(self):
        for file, inner_idx in zip(self.files, self.inner_idxs):
            df = pd.read_hdf(file)
            assert len(df) == len(inner_idx)

    def get_original_file_shapes(self):
        # assumes all files are hdf files
        file_shapes = []
        for file in self.files:
            with pd.HDFStore(file, mode="r") as store:
                shape = store.get_storer(store.keys()[0]).shape
                nrows = shape[0]
                ncols = shape[1]
            file_shapes.append((nrows, ncols))

        return file_shapes

    def get_file_lengths(self):
        return [np.sum(inner_idx) for inner_idx in self.inner_idxs]

    def get(self, idx: Iterable[int] | Iterable[bool]) -> SCDatasetInfo:
        idx = np.array(idx)
        if idx.dtype == bool:
            idx = np.where(idx)[0]
        file_seps = np.cumsum([0] + self.get_file_lengths())
        file_original_lengths = [nrow for nrow, _ in self.get_original_file_shapes()]

        new_inner_idxs = []

        for inner_idx, start, end, file_original_length in zip(
            self.inner_idxs, file_seps[:-1], file_seps[1:], file_original_lengths
        ):
            new_inner_idx = np.full(file_original_length, False)
            current_idx = idx[(start <= idx) & (idx < end)]
            if len(current_idx) > 0:
                inner_idx_int = np.where(inner_idx)[0]
                new_inner_idx[inner_idx_int[current_idx - start]] = True

            new_inner_idxs.append(new_inner_idx)

        return SCDatasetInfo(self.files, new_inner_idxs)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.get(range(*idx.indices(len(self))))
        elif isinstance(idx, Iterable):
            return self.get(idx)
        else:
            return self.get([idx])

    def __len__(self):
        return sum([np.sum(inner_idx) for inner_idx in self.inner_idxs])

    def __repr__(self):
        return f"SCDatasetInfo(files={self.files}, length={len(self)})"


def generate_tt_dataset(
    seed: int, n_3b: int, n_all4b: int, signal_ratio: float, test_ratio: float
) -> tuple[DatasetInfo, DatasetInfo]:
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


def generate_mother_dataset(
    n_3b: int,
    ratio_4b: float,
    signal_ratio: float,
    signal_filename: str,
    seed: int,
) -> tuple[SCDatasetInfo, pd.DataFrame]:
    """
    Generates "Mother samples" or all the samples used for a single experiment.
    Returns SCDataInfo and the DataFrame containing all the samples.

    1. Select n_3b 3b samples.
    2. Calculate total weight of 3b samples (=total_w_3b)
    3. Calculate total weight of bg4b and signal samples
    total_w_bg4b = ratio_4b / (1 - ratio_4b) * (1-signal_ratio) * total_w_3b,
    total_w_signal = ratio_4b / (1 - ratio_4b) * signal_ratio * total_w_3b.
    4. Choose bg4b samples in a random order until total weight of bg4b samples exceeds total_w_bg4b.
    5. Do the same for signal samples until total weight of signal samples exceeds total_w_signal.

    """
    assert 0 < n_3b
    assert 0 < ratio_4b < 1
    assert 0 <= signal_ratio <= 1

    directory = pathlib.Path("../events/MG3")
    np.random.seed(seed)

    dir_3b = directory / "dataframes" / "threeTag_picoAOD.h5"
    dir_bg4b = directory / "dataframes" / "fourTag_10x_picoAOD.h5"
    dir_signal = directory / "dataframes" / signal_filename

    # 1. Select n_3b 3b samples.
    df_3b = pd.read_hdf(dir_3b)
    assert n_3b <= len(df_3b)
    indices_3b = np.full(len(df_3b), False)
    indices_3b[np.random.choice(len(df_3b), n_3b, replace=False)] = True
    df_3b = df_3b[indices_3b]
    df_3b["signal"] = False

    # 2. Calculate total weight of 3b samples
    total_w_3b = df_3b["weight"].sum()

    # 3. Calculate total weight of bg4b and signal samples
    total_w_bg4b = ratio_4b / (1 - ratio_4b) * (1 - signal_ratio) * total_w_3b
    total_w_signal = ratio_4b / (1 - ratio_4b) * signal_ratio * total_w_3b

    # 4. Choose bg4b samples in a random order until total weight of bg4b samples exceeds total_w_bg4b.
    df_bg4b = pd.read_hdf(dir_bg4b)
    random_idx = np.random.permutation(len(df_bg4b))
    weights_bg4b = df_bg4b["weight"].values
    exceeded = np.cumsum(weights_bg4b[random_idx]) >= total_w_bg4b
    if not np.any(exceeded):
        raise ValueError("Not enough bg4b samples to fit the ratio.")
    idx_until = 0 if np.all(exceeded) else np.argmax(exceeded) + 1
    random_idx = np.sort(random_idx[:idx_until])
    indices_bg4b = np.full(len(df_bg4b), False)
    indices_bg4b[random_idx] = True
    df_bg4b = df_bg4b.iloc[random_idx]
    df_bg4b["signal"] = False

    # 5. Do the same for signal samples until total weight of signal samples exceeds total_w_signal.
    df_signal = pd.read_hdf(dir_signal)
    random_idx = np.random.permutation(len(df_signal))
    weights_signal = df_signal["weight"].values
    exceeded = np.cumsum(weights_signal[random_idx]) >= total_w_signal
    if not np.any(exceeded):
        raise ValueError("Not enough signal samples to fit the ratio.")
    idx_until = 0 if np.all(exceeded) else np.argmax(exceeded) + 1
    random_idx = np.sort(random_idx[:idx_until])
    indices_signal = np.full(len(df_signal), False)
    indices_signal[random_idx] = True
    df_signal = df_signal.iloc[random_idx]
    df_signal["signal"] = True

    inner_idxs = [indices_3b, indices_bg4b, indices_signal]
    df_list = [df_3b, df_bg4b, df_signal]
    df_list = [df for df in df_list if len(df) > 0]
    df = pd.concat(df_list).reset_index(drop=True)
    files = [dir_3b, dir_bg4b, dir_signal]

    scdinfo = SCDatasetInfo(files, inner_idxs)

    return scdinfo, df


def split_scdinfo(
    scdinfo: SCDatasetInfo,
    frac: float,
    seed: int,
) -> tuple[SCDatasetInfo, SCDatasetInfo]:
    """
    Splits SCDatasetInfo into two SCDatasetInfo objects.
    """
    assert 0 < frac < 1

    np.random.seed(seed)
    n_total = len(scdinfo)
    n_1 = int(frac * n_total)

    idx = np.random.permutation(n_total)
    idx_1 = idx[:n_1]
    idx_2 = idx[n_1:]

    scdinfo_1 = scdinfo[idx_1]
    scdinfo_2 = scdinfo[idx_2]

    return scdinfo_1, scdinfo_2


def split_scdinfo_multiple(
    scdinfo: SCDatasetInfo,
    fracs: list[float],
    seed: int,
) -> list[SCDatasetInfo]:
    """
    Splits SCDatasetInfo into multiple SCDatasetInfo objects.
    """

    assert all([0 < frac < 1 for frac in fracs]), "fracs must be between 0 and 1"
    assert np.isclose(sum(fracs), 1), f"sum of fracs is {sum(fracs)}, not 1"

    n_total = len(scdinfo)
    frac_cumsum = np.cumsum(fracs)
    idxs = [int(frac * n_total) for frac in frac_cumsum[:-1]]
    idxs = [0] + idxs + [n_total]

    assert len(idxs) == len(fracs) + 1
    assert idxs[-1] == n_total

    np.random.seed(seed)
    random_idx = np.random.permutation(n_total)

    scdinfos = []
    for i in range(len(fracs)):
        scdinfos.append(scdinfo[random_idx[idxs[i] : idxs[i + 1]]])

    return scdinfos


class MotherSamples:
    SAVE_DIR = pathlib.Path(__file__).parent / "data/mother_samples"
    META_DIR = pathlib.Path(__file__).parent / "data/metadata/mother_samples.pkl"

    def __init__(self, scdinfo: SCDatasetInfo, hash: str, hparams: dict):
        self._scdinfo = scdinfo
        self._hash = hash
        self._hparams = hparams

    @property
    def scdinfo(self):
        return self._scdinfo

    @property
    def hash(self):
        return self._hash

    @property
    def hparams(self):
        return self._hparams

    @staticmethod
    def from_hparams(hparams: dict) -> MotherSamples:
        require_keys(
            hparams, ["n_3b", "ratio_4b", "signal_ratio", "signal_filename", "seed"]
        )
        scdinfo, _ = generate_mother_dataset(
            hparams["n_3b"],
            hparams["ratio_4b"],
            hparams["signal_ratio"],
            hparams["signal_filename"],
            hparams["seed"],
        )
        name = create_hash(MotherSamples.SAVE_DIR)

        return MotherSamples(scdinfo, name, hparams)

    def save(self, verbose=True):
        if verbose:
            print(f"Saving Mother Samples: {self.hash}")
        with open(MotherSamples.SAVE_DIR / self.hash, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(hash: str) -> MotherSamples:
        with open(MotherSamples.SAVE_DIR / hash, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_metadata() -> dict[str, dict]:
        with open(MotherSamples.META_DIR, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def update_metadata():
        with open(MotherSamples.META_DIR, "wb") as f:
            hashes, hparams = MotherSamples.find(
                {}, return_hparams=True, from_metadata=False
            )
            pickle.dump(dict(zip(hashes, hparams)), f)

    @staticmethod
    def find(
        hparam_filter: dict[str, any],
        return_hparams=False,
        sort_by: list[str] = [],
        from_metadata=True,
    ) -> list[str]:

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
            all_hash_hparams = MotherSamples.load_metadata()
            for hash_, hparams in all_hash_hparams.items():
                if is_match(hparams):
                    hash_hparams.append((hash_, hparams))
        else:
            for file in tqdm.tqdm(MotherSamples.SAVE_DIR.glob("*")):
                tinfo = MotherSamples.load(file)
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

    def __repr__(self) -> str:
        return f"MotherSamples(hash={self.hash}, hparams={self.hparams})"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return len(self.scdinfo)
