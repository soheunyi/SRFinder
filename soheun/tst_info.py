from __future__ import annotations
import pathlib
import pickle

import numpy as np
import tqdm

from dataset import SCDatasetInfo
from training_info import TrainingInfoV2
from utils import create_hash

# get current directory of the file
TST_SAVE_DIR = pathlib.Path(__file__).parent / "data/tst_info"
# check if TST_SAVE_DIR exists, if not, create it
TST_SAVE_DIR.mkdir(parents=True, exist_ok=True)
TST_META_DIR = pathlib.Path(__file__).parent / "data/metadata/tst_info.pkl"


class TSTInfo:
    SAVE_DIR = TST_SAVE_DIR
    META_DIR = TST_META_DIR

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
        self._hash = create_hash(TSTInfo.SAVE_DIR)
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
        with open(TSTInfo.SAVE_DIR / self.hash, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(hash: str) -> TSTInfo:
        with open(TSTInfo.SAVE_DIR / hash, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def get_existing_hparams():
        _, hparams = TSTInfo.find({}, return_hparams=True)
        return hparams

    @staticmethod
    def load_metadata() -> dict[str, dict]:
        with open(TSTInfo.META_DIR, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def update_metadata():
        with open(TSTInfo.META_DIR, "wb") as f:
            hashes, hparams = TSTInfo.find({}, return_hparams=True, from_metadata=False)
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
            all_hash_hparams = TSTInfo.load_metadata()
            for hash_, hparams in all_hash_hparams.items():
                if is_match(hparams):
                    hash_hparams.append((hash_, hparams))
        else:
            for file in tqdm.tqdm(TSTInfo.SAVE_DIR.glob("*")):
                tinfo = TSTInfo.load(file)
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
