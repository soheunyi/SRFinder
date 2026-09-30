"""Score export alignment, complete coverage and raw float32 preservation."""
import hashlib
import pathlib
import sys
import tempfile
import numpy as np
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from artifacts.training_store import TrainingStore
from artifacts.export_scores import export_member_scores


def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Invalid prediction stream accepted')


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as temporary:
        root=pathlib.Path(temporary)
        store=TrainingStore(root/'store')
        dataset=store.put_dataset(hashlib.sha256(b'source').hexdigest(),5,{'version':1})
        ids=np.array([4,0,2,1,3],dtype=np.int64)
        split=store.put_split(dataset,'X2',ids,{'algorithm':'fixture','version':1,'seed':0})
        model=torch.nn.Linear(2,2)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.,0.],[1.,2.]]))
            model.bias.zero_()
        weights=root/'best.pt'
        torch.save(model.state_dict(),weights)
        owner=store.put_model(weights,{'member':0},{'dtype':'float32'},[split])
        features=torch.tensor([[1000.,1000.],[1.,1.],[2.,1.],[0.,0.],[-1000.,-1000.]])
        factory=lambda record:torch.nn.Linear(2,2)
        batches=[(ids[:2],features[:2]),(ids[2:],features[2:])]
        result=export_member_scores(store,owner,split,ids,batches,factory)
        actual=store.load_array(result)
        np.testing.assert_array_equal(actual,np.array([3000.,3.,4.,0.,-3000.],dtype=np.float32))
        before=store.storage_stats()
        def forbidden_factory(record):
            raise AssertionError('Cache hit unexpectedly reconstructed the model')
        assert export_member_scores(store,owner,split,ids,iter(()),forbidden_factory)==result
        assert export_member_scores(store,owner,split,ids,batches,factory,reuse_existing=False)==result
        assert store.storage_stats()==before
        rejects(lambda:export_member_scores(store,owner,split,ids,[(ids[::-1],features)],factory,reuse_existing=False))
        rejects(lambda:export_member_scores(store,owner,split,ids,batches[:1],factory,reuse_existing=False))
        rejects(lambda:export_member_scores(store,owner,split,ids,[(ids,features.double())],factory,reuse_existing=False))
        broken=features.clone();broken[0,0]=float('nan')
        rejects(lambda:export_member_scores(store,owner,split,ids,[(ids,broken)],factory,reuse_existing=False))
        assert store.storage_stats()==before,'Invalid stream published an artifact'
        cached = store.read(result)['identity']['inference_recipe']
        assert store.find_scores(owner,split,'log_density_ratio',cached)==result
        assert store.find_scores(owner,split,'log_density_ratio',{**cached,'version':2}) is None
        store.payload_path(result).write_bytes(b'broken cache')
        rejects(lambda:export_member_scores(store,owner,split,ids,iter(()),forbidden_factory))
    print('PASS: raw unclipped float32 export, model loading, row alignment, coverage, immutable retry')


if __name__=='__main__':
    main()
