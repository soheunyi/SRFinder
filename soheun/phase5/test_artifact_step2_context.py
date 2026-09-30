"""New encoder artifact -> existing smeared-feature path, with no old model lookup."""
import hashlib
import pathlib
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from artifacts.training_store import TrainingStore
from artifacts.model_loading import model_from_record,load_model
from artifacts.step2_context import ArtifactStep2Context
from training_info import TrainingInfo
from constants import FEATURES
from independent_data import stream_identity


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        root=pathlib.Path(tmp);store=TrainingStore(root/'store')
        dataset_recipe={'seed':0,'signal_ratio':0.,'n_3b':384,'ratio_4b':.5,
                        'signal_filename':'fixture.h5','base_fvt_train_ratio':.5}
        base_hp={'step':1,'model':'FvTClassifier','model_seed':0,'dataset':dataset_recipe,
                 'dim_dijet_features':6,'dim_quadjet_features':6,'depth':{'encoder':4,'decoder':1}}
        model=model_from_record({'kind':'model','id':'fixture','identity':{'training_recipe':{'hparams':base_hp}}})
        weights=root/'best.pt';torch.save(model.state_dict(),weights)
        dataset=store.put_dataset(hashlib.sha256(b'fixture').hexdigest(),768,dataset_recipe)
        split=store.put_split(dataset,'X1',np.arange(384,dtype=np.int64),{'algorithm':'fixture','version':1,'seed':0})
        owner=store.put_model(weights,{'member':0},{'hparams':base_hp},[split])
        x=torch.rand(384,4,4)
        x[:,0]=40+100*x[:,0];x[:,1]=2*x[:,1]-1;x[:,2]=6*x[:,2]-3;x[:,3]=5+15*x[:,3]
        frame=pd.DataFrame(x.reshape(384,16).numpy(),columns=FEATURES)
        frame['fourTag']=np.arange(384)%2;frame['weight']=1.
        source=SimpleNamespace(hparams=base_hp,ms_hash='fixture',ms_idx=np.arange(768)<384,
            scdinfo=SimpleNamespace(fetch_data=lambda:frame.copy()))
        hp={'experiment_name':'fixture_step2','model':'AttentionClassifier','depth':8,
            'dim_quadjet_features':6,'model_seed':0,'train_seed':0,'data_seed':0,'val_ratio':.33,
            'fit_batch_size':1024,'smearing':{'noise_scale':2.,'seed':0,'hard_cutoff':False,'scale_mode':'std'}}
        context=ArtifactStep2Context(store,owner,source,dataset,hp)
        with patch.object(TrainingInfo,'load',side_effect=AssertionError('Legacy model lookup')):
            actual=context.fetch_train_val_smeared_features(FEATURES,'fourTag','weight',device='cpu',training_alignment=32,retain_validation=True)
        reference=TrainingInfo(context.hparams,ms_hash='fixture',ms_idx=source.ms_idx)
        with patch.object(TrainingInfo,'scdinfo',property(lambda self:source.scdinfo)), \
             patch.object(TrainingInfo,'base_fvt_model',property(lambda self:load_model(store,owner))):
            expected=reference.fetch_train_val_smeared_features(FEATURES,'fourTag','weight',device='cpu',training_alignment=32,retain_validation=True)
        for a,b in zip(actual,expected):
            for u,v in zip(a.tensors,b.tensors):torch.testing.assert_close(u,v,rtol=0,atol=0)
        identity=stream_identity(context.hparams)
        changed=dict(context.hparams,smearing={**hp['smearing'],'noise_scale':1.})
        assert stream_identity(changed)!=identity
        assert stream_identity(dict(context.hparams,encoder_mode='last'))!=identity
        assert stream_identity(dict(context.hparams,dataloader={'preload_to_gpu':True,'num_workers':0}))==identity
        try:context.save()
        except RuntimeError:pass
        else:raise AssertionError('Legacy cache write was allowed')
    print('PASS: new encoder binding, exact smeared features, transform-aware identities and legacy-write rejection')


if __name__=='__main__':main()
