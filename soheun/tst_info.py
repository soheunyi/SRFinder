from __future__ import annotations
import pathlib
import random
import string
import pickle
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import tqdm

from dataset import SCDatasetInfo
from training_info import TrainingInfoV2, create_hash

# get current directory of the file
TST_SAVE_DIR = pathlib.Path(__file__).parent / "data/tst_info"
# check if TST_SAVE_DIR exists, if not, create it
TST_SAVE_DIR.mkdir(parents=True, exist_ok=True)


class TSTInfo:
    def __init__(
        self,
        hparams: dict,
        scdinfo_tst: SCDatasetInfo,
        SR_stats: np.ndarray,
        SR_cut: float,
        CR_cut: float,
        base_fvt_tinfo_hash: TrainingInfoV2,
        CR_fvt_tinfo_hash: TrainingInfoV2,
    ):
        """
        Save the hparams, for two sample test.
        """
        self._hparams = hparams
        self._hash = create_hash(TST_SAVE_DIR)
        self._aux_info = {}
        self.scdinfo_tst = scdinfo_tst
        self.SR_stats = SR_stats
        self.SR_cut = SR_cut
        self.CR_cut = CR_cut
        self.base_fvt_tinfo_hash = base_fvt_tinfo_hash
        self.CR_fvt_tinfo_hash = CR_fvt_tinfo_hash

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
        print(f"Saving TSTInfo with hash: {self.hash}")
        with open(TST_SAVE_DIR / self.hash, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(hash: str) -> TSTInfo:
        with open(TST_SAVE_DIR / hash, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def get_existing_hparams():
        hparams_list = []
        for file in TST_SAVE_DIR.glob("*"):
            tinfo = TSTInfo.load(file)
            hparams_list.append(tinfo.hparams)

        return hparams_list

    @staticmethod
    def find(hparam_filter: dict[str, any], return_hparams=False) -> list[str]:
        hashes = []
        for file in tqdm.tqdm(TST_SAVE_DIR.glob("*")):
            tinfo = TSTInfo.load(file)
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
