"""Verify new-region CR-only selection and existing member-specific split behavior."""
import hashlib
import pathlib
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from artifacts.training_store import TrainingStore,canonical
from artifacts.source_context import verify_source_context
from artifacts.regions import define_regions,classify_X2
from artifacts.step3_context import ArtifactStep3Context
from dataset import SCDatasetInfo,MotherSamples
from training_info import TrainingInfo


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        root=pathlib.Path(tmp);path=root/'pool.h5'
        frame=pd.DataFrame({'x':np.arange(512,dtype=np.float32),'fourTag':np.arange(512)%2,'weight':np.ones(512)})
        frame.to_hdf(path,key='df')
        mask=np.ones(512,dtype=bool);raw=SCDatasetInfo([path],[mask])
        params={'seed':3,'n_3b':256,'ratio_4b':.5,'signal_ratio':0.,'signal_filename':'pool.h5','base_fvt_train_ratio':.5}
        source_hp={'step':1,'model':'FvTClassifier','dataset':params,'data_seed':0,'val_ratio':.33}
        desc={'pools':[{'name':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}],
              'mother_parameters':params,'mother_selection_sha256':[hashlib.sha256(mask.tobytes()).hexdigest()]}
        store=TrainingStore(root/'store');dataset=store.put_dataset(hashlib.sha256(canonical(desc)).hexdigest(),512,desc)
        source=verify_source_context(store,dataset,source_hp,raw,source_root=root)
        splits=[store.put_split(dataset,d,source.indices(d),{'algorithm':'fixture_outer','version':1,'seed':3}) for d in ('X1','X2')]
        weights=root/'weights';weights.write_bytes(b'opaque fixture')
        base=store.put_model(weights,{'member':0},{'hparams':source_hp},[splits[0]])
        smooth=store.put_model(weights,{'member':0},{'hparams':{'step':2,'model':'AttentionClassifier','encoder_hash':'artifact:'+base}},[splits[0]])
        bp=[];sp=[]
        for d,split in zip(('X1','X2'),splits):
            idx=source.indices(d)
            bp.append(store.put_scores(base,split,(idx/512).astype(np.float32),'log_density_ratio'))
            sp.append(store.put_scores(smooth,split,np.zeros(len(idx),dtype=np.float32),'log_density_ratio'))
        x1=source.indices('X1');events=np.empty(len(x1),dtype=[('is_4b','?'),('weight','<f8')])
        events['is_4b']=frame.fourTag.values[x1];events['weight']=1.
        eid=store.put_array('event_metadata',{'dataset_id':dataset,'split_id':splits[0]},events)
        region=define_regions(store,[bp[0]],eid,[sp[0]],sr_fraction=.2,cr_fraction=.8,quantile_recipe='sr_quantile_cr_complement_v2')
        selected=classify_X2(store,region,[bp[1]],[sp[1]])
        allowed=set(source.indices('X2')[selected['CR']].tolist())
        contexts=[]
        for member in (0,1):
            hp={'experiment_name':'fixture_CR','model':'FvTClassifier','model_seed':member,
                'data_seed':member,'train_seed':member,'val_ratio':.33,'fit_batch_size':1024}
            context=ArtifactStep3Context(store,region,source,[bp[1]],[sp[1]],hp)
            contexts.append(context)
            assert set(np.flatnonzero(context.ms_idx))==allowed
            assert not np.any(context.ms_idx & source.ms_idx)
            actual=context.fetch_train_val_tensor_datasets(['x'],'fourTag','weight',training_alignment=32,retain_validation=True)
            reference=TrainingInfo(context.hparams,source.ms_hash,context.ms_idx)
            with patch.object(MotherSamples,'load',return_value=source.mother_samples):
                expected=reference.fetch_train_val_tensor_datasets(['x'],'fourTag','weight',training_alignment=32,retain_validation=True)
            for a,b in zip(actual,expected):
                for u,v in zip(a.tensors,b.tensors):torch.testing.assert_close(u,v,rtol=0,atol=0)
                assert set(a.tensors[0][:,0].int().tolist())<=allowed
        assert contexts[0].hash!=contexts[1].hash
        try:contexts[0].save()
        except RuntimeError:pass
        else:raise AssertionError('Legacy write allowed')
    print('PASS: frozen-region CR-only rows, no X1/SR leakage, native member splits and stable distinct identities')


if __name__=='__main__':main()
