"""Round-trip real supported architectures without legacy TrainingInfo lookups."""
import hashlib
import pathlib
import sys
import tempfile
import numpy as np
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from artifacts.training_store import TrainingStore
from artifacts.model_loading import model_from_record,load_model
from artifacts.export_scores import export_member_scores


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        root=pathlib.Path(directory);store=TrainingStore(root/'store')
        dataset=store.put_dataset(hashlib.sha256(b'fixture').hexdigest(),64,{'version':1})
        ids=np.arange(64,dtype=np.int64)
        split=store.put_split(dataset,'X2',ids,{'algorithm':'fixture','version':1,'seed':0})
        for kind in ('FvTClassifier','AttentionClassifier'):
            hp={'model':kind,'model_seed':13,'step':1 if kind=='FvTClassifier' else 2,
                'dim_dijet_features':6,'dim_quadjet_features':6,
                'depth':{'encoder':4,'decoder':1} if kind=='FvTClassifier' else 8,'repr_norm':False}
            template={'kind':'model','id':'fixture','identity':{'training_recipe':{'hparams':hp}}}
            original=model_from_record(template).eval()
            path=root/f'{kind}.pt';torch.save(original.state_dict(),path)
            owner=store.put_model(path,{'member':13,'kind':kind},{'hparams':hp},[split])
            before=torch.random.get_rng_state().clone()
            loaded=load_model(store,owner)
            assert torch.equal(before,torch.random.get_rng_state()),'Loading altered caller RNG'
            if kind=='FvTClassifier':
                x=torch.rand(64,4,4)
                x[:,0]=40+100*x[:,0];x[:,1]=2*x[:,1]-1;x[:,2]=6*x[:,2]-3;x[:,3]=5+15*x[:,3]
                x=x.reshape(64,16)
            else:x=torch.randn(64,6,3)
            with torch.inference_mode():
                expected=original(x)
                torch.testing.assert_close(loaded(x),expected,rtol=0,atol=0)
            score=export_member_scores(store,owner,split,ids,[(ids,x)])
            np.testing.assert_array_equal(store.load_array(score),(expected[:,1]-expected[:,0]).numpy())
    print('PASS: FvT and depth-8 attention artifact loading, caller RNG preservation and default score export')


if __name__=='__main__':main()
