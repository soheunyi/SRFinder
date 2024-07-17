from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from typing import Iterable, Self
from torch.utils.data import TensorDataset

from fvt_representations import get_fvt_reprs
from fvt_classifier import FvTClassifier


class EventsData:
    def __init__(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        is_4b: np.ndarray,
        is_signal: np.ndarray,
        name: str,
        npd: dict[str, np.ndarray] = {},
    ):
        assert isinstance(X, np.ndarray) and X.ndim == 2
        assert isinstance(weights, np.ndarray) and weights.ndim == 1
        assert isinstance(is_4b, np.ndarray) and is_4b.ndim == 1
        assert isinstance(is_signal, np.ndarray) and is_signal.ndim == 1
        assert X.shape[0] == weights.shape[0] == is_4b.shape[0] == is_signal.shape[0]

        self._X = X
        self._weights = weights
        self._is_4b = is_4b
        self._is_signal = is_signal

        self._is_3b = None
        self._is_bg4b = None

        self.fvt_score = None
        self.view_score = None
        self.q_repr = None

        self._name = name
        self._npd = npd

        self.update()

    def update_npd(self, key: str, data: np.ndarray):
        assert len(data) == len(self)
        self._npd[key] = data

    def update(self):
        self._is_3b = ~self._is_4b
        self._is_bg4b = self._is_4b & ~self._is_signal

    def set_model_scores(self, model: FvTClassifier):
        device = model.device

        fvt_score = model.predict(self.X_torch)[:, 1].cpu().detach().numpy()
        q_repr, view_score = get_fvt_reprs(self.X_torch, model, device=device)

        self.fvt_score = fvt_score
        self.view_score = view_score
        self.q_repr = q_repr

    def to_tensor_dataset(self) -> TensorDataset:
        return TensorDataset(self.X_torch, self.is_4b_torch, self.weights_torch)

    @staticmethod
    def merge(events_data_list: list[EventsData]):
        X = np.concatenate([data.X for data in events_data_list], axis=0)
        weights = np.concatenate([data.weights for data in events_data_list], axis=0)
        is_4b = np.concatenate([data.is_4b for data in events_data_list], axis=0)
        is_signal = np.concatenate(
            [data.is_signal for data in events_data_list], axis=0
        )
        name = "_".join([data.name for data in events_data_list])
        npd = {
            key: np.concatenate([data._npd[key] for data in events_data_list], axis=0)
            for key in events_data_list[0]._npd.keys()
        }

        return EventsData(X, weights, is_4b, is_signal, name, npd)

    @staticmethod
    def from_dataframe(df: pd.DataFrame, features, name: str = "") -> EventsData:
        assert isinstance(df, pd.DataFrame)
        assert all([f in df.columns for f in features])
        assert (
            "weight" in df.columns
            and "fourTag" in df.columns
            and "signal" in df.columns
        )
        X = df.loc[:, features].values
        weights = df["weight"].values
        is_4b = df["fourTag"].values
        is_signal = df["signal"].values

        # no npd
        return EventsData(X, weights, is_4b, is_signal, name)

    def reweight(self, new_weights: np.ndarray):
        assert isinstance(new_weights, np.ndarray) and new_weights.ndim == 1
        assert len(new_weights) == len(self)

        self._weights = new_weights

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.permutation(self._X.shape[0])
        self._X = self._X[idx]
        self._weights = self._weights[idx]
        self._is_4b = self._is_4b[idx]
        self._is_signal = self._is_signal[idx]

        for key in self._npd.keys():
            self._npd[key] = self._npd[key][idx]

        self.update()

    def split(
        self,
        frac: float,
        name_1: str = None,
        name_2: str = None,
        seed: int = None,
    ) -> tuple[EventsData, EventsData]:
        assert 0 <= frac <= 1
        if seed is not None:
            np.random.seed(seed)
        self.shuffle(seed=seed)

        n = int(frac * len(self))

        npd_1 = {key: data[:n] for key, data in self._npd.items()}
        npd_2 = {key: data[n:] for key, data in self._npd.items()}

        data1 = EventsData(
            self._X[:n],
            self._weights[:n],
            self._is_4b[:n],
            self._is_signal[:n],
            self._name + "_1" if name_1 is None else name_1,
            npd_1,
        )
        data2 = EventsData(
            self._X[n:],
            self._weights[n:],
            self._is_4b[n:],
            self._is_signal[n:],
            self._name + "_2" if name_2 is None else name_2,
            npd_2,
        )

        return data1, data2

    def fit_batch_size(self, batch_size: int):
        n_batches = len(self) // batch_size
        n_new = n_batches * batch_size

        self.trim(n_new)

    def trim(self, n: int):
        assert 0 <= n <= len(self)

        self._X = self._X[:n]
        self._weights = self._weights[:n]
        self._is_4b = self._is_4b[:n]
        self._is_signal = self._is_signal[:n]

        for key in self._npd.keys():
            self._npd[key] = self._npd[key][:n]

        self.update()

    def get(self, idx: Iterable[int], name: str = "") -> EventsData:
        events = EventsData(
            self._X[idx],
            self._weights[idx],
            self._is_4b[idx],
            self._is_signal[idx],
            name,
            {key: data[idx] for key, data in self._npd.items()},
        )

        if self.fvt_score is not None:
            events.fvt_score = self.fvt_score[idx]
        if self.view_score is not None:
            events.view_score = self.view_score[idx]
        if self.q_repr is not None:
            events.q_repr = self.q_repr[idx]

        return events

    def get_from_mask(self, mask: np.ndarray, name: str = "") -> EventsData:
        return self.get(np.where(mask)[0], name)

    def filter(self, mask_str: str, name: str = "") -> EventsData:
        assert mask_str in self.npd.keys()
        mask = self.npd[mask_str]
        return self.get_from_mask(mask, name)

    def get_signal(self) -> Self:
        return self.get(self._is_signal, "signal")

    def get_bg4b(self) -> Self:
        return self.get(self._is_bg4b, "bg4b")

    def get_4b(self) -> Self:
        return self.get(self._is_4b, "4b")

    def get_3b(self) -> Self:
        return self.get(self._is_3b, "3b")

    def __repr__(self):
        ratio_4b = self.total_weight_4b / self.total_weight
        ratio_signal = self.total_weight_signal / self.total_weight_4b
        return f"EventsData(name={self._name}, w_sum={self.total_weight:.2f}, ratio_4b={ratio_4b:.2f}, ratio_signal={ratio_signal:.2f})"

    def __len__(self):
        return self._X.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.get(range(*idx))
        elif isinstance(idx, Iterable):
            return self.get(idx)
        else:
            return self.get([idx])

    @property
    def total_weight(self):
        return np.sum(self._weights)

    @property
    def total_weight_4b(self):
        return np.sum(self._weights[self._is_4b])

    @property
    def total_weight_signal(self):
        return np.sum(self._weights[self._is_signal])

    @property
    def X(self):
        return self._X

    @property
    def weights(self):
        return self._weights

    @property
    def is_4b(self):
        return self._is_4b

    @property
    def is_signal(self):
        return self._is_signal

    @property
    def is_3b(self):
        return self._is_3b

    @property
    def is_bg4b(self):
        return self._is_bg4b

    @property
    def is_signal(self):
        return self._is_signal

    @property
    def name(self):
        return self._name

    @property
    def npd(self):
        return self._npd

    @property
    def X_torch(self) -> torch.Tensor:
        return torch.tensor(self._X, dtype=torch.float32)

    @property
    def weights_torch(self) -> torch.Tensor:
        return torch.tensor(self._weights, dtype=torch.float32)

    @property
    def is_4b_torch(self) -> torch.Tensor:
        return torch.tensor(self._is_4b, dtype=torch.long)

    @property
    def att_q_repr(self) -> np.ndarray:
        assert self.q_repr is not None
        assert self.view_score is not None

        return (self.q_repr @ self.view_score[:, :, None]).reshape(
            -1, self.q_repr.shape[1]
        )
