"""Verify registered raw pools/mother selection, then reconstruct X1/X2 in memory."""
from copy import deepcopy
from functools import cached_property
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .training_store import canonical


def _file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


class VerifiedSourceContext:
    def __init__(self,dataset_id,hparams,full_source):
        self.dataset_id=dataset_id
        self._hparams=deepcopy(hparams)
        self.full_source=full_source
        self.ms_hash='artifact-source:'+dataset_id
        n=len(full_source)
        ratio=float(hparams['dataset']['base_fvt_train_ratio'])
        if not 0<ratio<1:raise ValueError('Invalid X1 fraction')
        mask=np.zeros(n,dtype=bool);mask[:int(n*ratio)]=True
        np.random.RandomState(int(hparams['dataset']['seed'])).shuffle(mask)
        mask.setflags(write=False)
        self.ms_idx=mask

    @property
    def hparams(self):return deepcopy(self._hparams)

    @property
    def mother_samples(self):
        return SimpleNamespace(scdinfo=self.full_source,hparams=deepcopy(self._hparams['dataset']),hash=self.ms_hash)

    @cached_property
    def scdinfo(self):return self.full_source[self.ms_idx]

    @cached_property
    def X2_scdinfo(self):return self.full_source[~self.ms_idx]

    def indices(self,domain):
        if domain not in ('X1','X2'):raise ValueError('Unknown source domain')
        return np.flatnonzero(self.ms_idx if domain=='X1' else ~self.ms_idx).astype(np.int64)


def verify_source_context(store,dataset_id,hparams,mother_scdinfo,*,source_root,fingerprints=None):
    """mother_scdinfo may come from a shared cache or seed-based reconstruction.

    Verify content before using it. Source paths are resolved independently of
    cwd; no source file, selection mask or legacy metadata is written.
    """
    from dataset import SCDatasetInfo
    record=store.read(dataset_id,'dataset')
    expected=record['identity']['selection_recipe']
    required={'pools','mother_parameters','mother_selection_sha256'}
    if not required<=expected.keys():raise ValueError('Unsupported source descriptor')
    if hparams.get('dataset')!=expected['mother_parameters']:
        raise ValueError('Mother parameters differ from the registered source')
    paths=[(Path(p) if Path(p).is_absolute() else Path(source_root)/p).resolve() for p in mother_scdinfo.files]
    masks=[np.asarray(mask) for mask in mother_scdinfo.inner_idxs]
    if any(mask.dtype!=np.bool_ or mask.ndim!=1 for mask in masks):
        raise ValueError('Mother selections must be one-dimensional boolean masks')
    actual={'pools':[{'name':p.name,'sha256':fingerprints.sha256(p) if fingerprints is not None else _file_sha(p)} for p in paths],
            'mother_parameters':deepcopy(hparams['dataset']),
            'mother_selection_sha256':[hashlib.sha256(mask.tobytes()).hexdigest() for mask in masks]}
    if actual!=expected or hashlib.sha256(canonical(actual)).hexdigest()!=record['identity']['source_fingerprint']:
        raise ValueError('Raw pools or mother selection differ from the registered dataset')
    source=SCDatasetInfo(paths,[mask.copy() for mask in masks])
    if len(source)!=record['identity']['row_count']:
        raise ValueError('Registered row count differs from verified mother selection')
    return VerifiedSourceContext(dataset_id,hparams,source)
