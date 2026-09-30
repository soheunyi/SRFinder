"""In-memory Step-2 context using a new encoder artifact and validated source data.

This deliberately cannot be saved as a legacy TrainingInfo pickle. The source
context must already have been checked against its registered dataset version.
"""
from copy import deepcopy
import hashlib
from training_info import TrainingInfo
from .training_store import canonical
from .model_loading import load_model


class ArtifactStep2Context(TrainingInfo):
    def __init__(self, store, encoder_id, source_info, source_dataset_id, hparams):
        record=store.read(encoder_id,'model')
        base_hp=record['identity']['training_recipe'].get('hparams',{})
        if base_hp.get('model')!='FvTClassifier' or int(base_hp.get('step',0))!=1:
            raise ValueError('Step 2 requires a Step-1 FvT encoder artifact')
        datasets={store.read(key,'split')['identity']['dataset_id'] for key in record['identity']['split_ids']}
        if datasets!={source_dataset_id}:
            raise ValueError('Source dataset does not match the encoder training provenance')
        dataset=store.read(source_dataset_id,'dataset')
        source_hp=source_info.hparams
        if source_hp.get('dataset')!=base_hp.get('dataset') or int(source_hp.get('step',source_hp.get('aux_info_step',0)))!=1:
            raise ValueError('Source context does not match the Step-1 dataset recipe')
        if len(source_info.ms_idx)!=dataset['identity']['row_count']:
            raise ValueError('Source selection length differs from registered dataset')
        hp=deepcopy(hparams)
        if hp.get('model')!='AttentionClassifier':
            raise ValueError('Step 2 uses AttentionClassifier')
        hp.update(step=2,dataset=deepcopy(base_hp['dataset']),encoder_hash='artifact:'+encoder_id,
                  encoder_mode='best',source_dataset_id=source_dataset_id)
        self._artifact_store=store
        self._encoder_artifact=encoder_id
        self._source_info=source_info
        super().__init__(hp,ms_hash=source_info.ms_hash,ms_idx=source_info.ms_idx)
        self._hash='artifact-step2-'+hashlib.sha256(canonical(hp)).hexdigest()[:24]

    @property
    def scdinfo(self):
        return self._source_info.scdinfo

    @property
    def mother_samples(self):
        return self._source_info.mother_samples

    @property
    def base_fvt_model(self):
        return load_model(self._artifact_store,self._encoder_artifact)

    def save(self,*args,**kwargs):
        raise RuntimeError('Artifact-backed contexts must publish new artifacts, not legacy TrainingInfo pickles')
