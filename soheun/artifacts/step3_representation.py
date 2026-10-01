"""Single-upstream CR diagnostic on a frozen base encoder's representations."""
from copy import deepcopy
import torch
from torch.utils.data import TensorDataset
from .step3_context import ArtifactStep3Context
from .model_loading import load_model


class ArtifactStep3RepresentationContext(ArtifactStep3Context):
    model_types=('AttentionClassifier',)

    def __init__(self,store,region_id,source_info,base_X2_scores,smeared_X2_scores,hparams,encoder_id):
        definition=store.read(region_id,'region')['identity']['definition']
        if definition['base_models']!=[encoder_id]:
            raise ValueError('Representation diagnostic requires its single upstream region encoder')
        base=store.read(encoder_id,'model')['identity']['training_recipe']['hparams']
        if base.get('step')!=1 or base.get('model')!='FvTClassifier' or base.get('source_dataset_id')!=source_info.dataset_id:
            raise ValueError('Representation encoder stage/source differs')
        hp=deepcopy(hparams)
        width=int(hp.get('dim_q',hp.get('dim_quadjet_features',-1)))
        if width!=base['dim_quadjet_features'] or hp.get('dim_quadjet_features',width)!=width:
            raise ValueError('Representation width differs from encoder')
        if hp.get('smearing') is not None:raise ValueError('CR representation training does not smear features')
        hp.update(input_space='base_encoder',encoder_hash='artifact:'+encoder_id,encoder_mode='best',
                  dim_q=width,dim_quadjet_features=width,
                  representation_policy={'algorithm':'native_q_repr_after_member_split','version':1})
        self._store=store;self._encoder_id=encoder_id
        super().__init__(store,region_id,source_info,base_X2_scores,smeared_X2_scores,hp)

    def fetch_train_val_representation_datasets(self,features,label,weight,*,device='cpu',
                                               training_alignment=32,retain_validation=True):
        raw=self.fetch_train_val_tensor_datasets(features,label,weight,
            training_alignment=training_alignment,retain_validation=retain_validation)
        encoder=load_model(self._store,self._encoder_id,device=device)
        result=[]
        # Match the legacy diagnostic: split raw CR rows first, then q_repr.
        for data in raw:
            values=encoder.q_repr(data.tensors[0])
            if values.dtype!=torch.float32 or not torch.isfinite(values).all():
                raise ValueError('Invalid frozen-encoder representation')
            result.append(TensorDataset(values,*data.tensors[1:]))
        return tuple(result)
