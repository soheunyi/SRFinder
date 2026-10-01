"""In-memory Step-1 context over a verified, shared source selection."""
from copy import deepcopy
import hashlib
from training_info import TrainingInfo
from .training_store import canonical


class ArtifactStep1Context(TrainingInfo):
    def __init__(self,source_info,hparams):
        hp=deepcopy(hparams)
        if hp.get('model')!='FvTClassifier':raise ValueError('Step 1 uses FvTClassifier')
        hp.update(step=1,dataset=source_info.hparams['dataset'],source_dataset_id=source_info.dataset_id)
        self._source_info=source_info
        super().__init__(hp,ms_hash=source_info.ms_hash,ms_idx=source_info.ms_idx)
        self._hash='artifact-step1-'+hashlib.sha256(canonical(hp)).hexdigest()[:24]

    @property
    def scdinfo(self):return self._source_info.scdinfo

    @property
    def mother_samples(self):return self._source_info.mother_samples

    def save(self,*args,**kwargs):
        raise RuntimeError('Artifact-backed contexts must publish new artifacts, not legacy TrainingInfo pickles')
