"""In-memory original-feature CR training context bound to frozen new regions."""
from copy import deepcopy
import hashlib
import numpy as np
from training_info import TrainingInfo
from .training_store import canonical
from .regions import classify_X2


class ArtifactStep3Context(TrainingInfo):
    def __init__(self,store,region_id,source_info,base_X2_scores,smeared_X2_scores,hparams):
        region=store.read(region_id,'region')
        definition=region['identity']['definition']
        if source_info.dataset_id!=definition['dataset_id']:
            raise ValueError('Region and source dataset versions differ')
        result=classify_X2(store,region_id,base_X2_scores,smeared_X2_scores)
        split_id=store.read(base_X2_scores[0],'scores')['identity']['split_id']
        x2=source_info.indices('X2')
        store.verify_split(split_id,x2)
        if len(result['CR'])!=len(x2):raise ValueError('CR mask does not align with X2')
        mask=np.zeros(len(source_info.full_source),dtype=bool)
        mask[x2[result['CR']]]=True
        if not mask.any():raise ValueError('No CR training rows')
        hp=deepcopy(hparams)
        if hp.get('model')!='FvTClassifier':
            raise ValueError('This context is for original-feature CR FvT; representation diagnostics need their own binding')
        hp.update(step=3,dataset=source_info.hparams['dataset'],source_dataset_id=source_info.dataset_id,
            signal_region={'region_artifact_id':region_id,
                '4b_in_SR':region['identity']['requested_sr_fraction'],
                '4b_in_CR':region['identity']['requested_cr_fraction'],
                'ensemble_mode':definition['ensemble_mode'],
                'stats_type':'smeared' if definition['smeared_models'] else 'fvt',
                'SR_stats_hashes':['artifact:'+key for key in (definition['smeared_models'] or definition['base_models'])]})
        self._source_info=source_info
        self.region_id=region_id
        super().__init__(hp,ms_hash=source_info.ms_hash,ms_idx=mask)
        self._hash='artifact-step3-'+hashlib.sha256(canonical(hp)).hexdigest()[:24]

    @property
    def scdinfo(self):return self._source_info.full_source[self.ms_idx]

    @property
    def mother_samples(self):return self._source_info.mother_samples

    def save(self,*args,**kwargs):
        raise RuntimeError('Artifact-backed contexts must publish new artifacts, not legacy TrainingInfo pickles')
