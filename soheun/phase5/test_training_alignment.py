import pathlib,sys,types
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from training_info import TrainingInfo
from dataset import SCDatasetInfo

def frame(n):
    return pd.DataFrame({'x':np.arange(n,dtype=float),'fourTag':np.arange(n)%2,'weight':np.ones(n)})
class Fake:
    hparams={'fit_batch_size':1024,'encoder_hash':'fixture','encoder_mode':'best',
             'smearing':{'noise_scale':.1,'seed':0,'hard_cutoff':False,'scale_mode':'std'}}
    data_seed=17;val_ratio=.33
    fetch_train_val_data=TrainingInfo.fetch_train_val_data
    def fetch_train_val_scdinfo(self):return 'train','validation'

fake=Fake()
with patch.object(SCDatasetInfo,'fetch_multiple_data',return_value=[frame(2301),frame(1201)]):
    legacy=TrainingInfo.fetch_train_val_tensor_datasets(fake,['x'],'fourTag','weight')
    current=TrainingInfo.fetch_train_val_tensor_datasets(fake,['x'],'fourTag','weight',training_alignment=32,retain_validation=True)
    assert [len(d) for d in legacy]==[2048,1024]
    assert [len(d) for d in current]==[2272,1201]
    for dataset in current:assert torch.equal(dataset.tensors[1],dataset.tensors[0][:,0].long()%2)
    fake.aux_info={'data_batching_policy':{'training_alignment':32,'retain_validation':True}}
    restored=TrainingInfo.fetch_train_val_tensor_datasets(fake,['x'],'fourTag','weight')
    assert [len(d) for d in restored]==[2272,1201]
    fake.aux_info={}
class Encoder:
    def to(self,device):return self
    def representations(self,x):return (x.unsqueeze(-1),None)
fake.base_fvt_model=Encoder();fake.scdinfo=types.SimpleNamespace(fetch_data=lambda:frame(3502))
with patch('training_info.smear_features',side_effect=lambda x,**kw:x):
    legacy=TrainingInfo.fetch_train_val_smeared_features(fake,['x'],'fourTag','weight',device=torch.device('cpu'))
    current=TrainingInfo.fetch_train_val_smeared_features(fake,['x'],'fourTag','weight',device=torch.device('cpu'),training_alignment=32,retain_validation=True)
    assert [len(d) for d in legacy]==[2048,1024]
    assert [len(d) for d in current]==[2336,1156]
    for dataset in current:assert torch.equal(dataset.tensors[1],dataset.tensors[0][:,0,0].long()%2)
print('PASS: FvT and smeared-feature extraction retain 32-aligned training and all validation; legacy readers unchanged')
