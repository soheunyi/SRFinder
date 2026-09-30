"""Coverage, grouping, sample accounting and epoch-boundary resume regression."""
import argparse,copy,json,pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE.parent,HERE.parent/'phase3'):sys.path.insert(0,str(p))
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from independent_data import IndependentStackedDataModule
from stacked_fvt import StackedFvTClassifier
from stacked_attention_classifier import StackedAttentionClassifier
from fvt_classifier import FvTClassifier
from attention_classifier import AttentionClassifier
from resumable import ResumableIndividualSaver,StopAfterEpoch

LENGTHS=[96,160,416]
VAL_LENGTHS=[17,65,101]
SEEDS=[17,28,39]

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def stream_checks():
    train=[TensorDataset(torch.arange(n).float().view(-1,1)+i*10000,torch.zeros(n).long(),torch.ones(n)) for i,n in enumerate(LENGTHS)]
    val=[TensorDataset(torch.arange(n).float().view(-1,1)+i*10000,torch.zeros(n).long(),torch.ones(n)) for i,n in enumerate(VAL_LENGTHS)]
    def dm(ids,workers=0):
        return IndependentStackedDataModule([train[i] for i in ids],[val[i] for i in ids],64,
            shuffle_seeds=[SEEDS[i] for i in ids],estimator_ids=[str(i) for i in ids],num_workers=workers,
            batch_size_milestones=[1,3])
    for epoch in (0,1,2,3,4):
        original=None
        for ids,workers in [([0,1,2],0),([2,0,1],0),([0,1,2],2)]:
            module=dm(ids,workers)
            result={i:[] for i in ids}
            for batch in module.loader_for_epoch(epoch):
                for pos,b in enumerate(batch['members']):
                    if b is not None:
                        assert len(b[0])%32==0 and len(b[0])>=64
                        result[ids[pos]].append(b[0].flatten().tolist())
            for i in ids:
                flat=[v for batch in result[i] for v in batch]
                assert sorted(flat)==train[i].tensors[0].flatten().tolist()
            if original is None:original=result
            else:assert original==result
            restored=dm(ids,workers);restored.load_state_dict(module.state_dict())
            assert [b['members'][0][0].tolist() if b['members'][0] is not None else None for b in restored.loader_for_epoch(epoch)]==[b['members'][0][0].tolist() if b['members'][0] is not None else None for b in module.loader_for_epoch(epoch)]
        for i in range(3):
            solo=[b['members'][0][0].flatten().tolist() for b in dm([i]).loader_for_epoch(epoch)]
            assert solo==original[i]
        counts=[0,0,0]
        for batch in dm([0,1,2]).loader_for_epoch(epoch,False):
            for i,b in enumerate(batch['members']):
                if b is not None:counts[i]+=len(b[1])
        assert counts==VAL_LENGTHS
    class IndexedView:
        def __init__(self, dataset):
            self.dataset = dataset
            self.device = dataset.tensors[0].device
            self.indices = torch.arange(len(dataset)-1, -1, -1, device=self.device)
        def __len__(self): return len(self.indices)
        def gather(self, positions):
            ids = self.indices.index_select(0, positions)
            return tuple(t.index_select(0, ids) for t in self.dataset.tensors)
    indexed = IndependentStackedDataModule(train, [IndexedView(d) for d in val], 64, shuffle_seeds=SEEDS)
    probe = indexed.validation_probe(9)
    for i in range(3):
        assert torch.equal(probe.tensors[0][:,i], val[i].tensors[0].flip(0)[:9])
    partial=dm([0,1,2]);loader=partial.loader_for_epoch(0)
    partial.note_training_batch()
    try:partial.state_dict()
    except RuntimeError:pass
    else:raise AssertionError('Mid-epoch checkpoint incorrectly accepted')
    for _ in range(len(loader)-1):partial.note_training_batch()
    partial.state_dict()
    x=dm([0,1,2]);y=dm([2,1,0])
    try:y.load_state_dict(x.state_dict())
    except ValueError:pass
    else:raise AssertionError('Wrong identity order accepted')
    return {'all_rows_once':True,'alone_grouped_reordered_equal':True,'workers_0_vs_2_equal':True,
            'schedule_rebuild_idempotent':True,'tiny_tail_merged':True,'all_validation_rows':True,'mismatched_resume_rejected':True}

def datasets(kind):
    train=[];val=[]
    for i,n in enumerate(LENGTHS):
        gen=torch.Generator().manual_seed(100+i)
        def build(n):
            if kind=='fvt':
                x=torch.rand(n,4,4,generator=gen)
                x[:,0]=40+100*x[:,0];x[:,1]=2*x[:,1]-1;x[:,2]=6*x[:,2]-3;x[:,3]=5+15*x[:,3]
                x=x.reshape(n,16)
            else:x=torch.randn(n,6,3,generator=gen)
            return TensorDataset(x,torch.randint(2,(n,),generator=gen),torch.rand(n,generator=gen)+.5)
        train.append(build(n));val.append(build(VAL_LENGTHS[i]))
    return train,val

class Audit(pl.Callback):
    def __init__(self):self.rows=[]
    def on_validation_end(self,trainer,module):
        if trainer.sanity_checking:return
        members=getattr(module,'fvt_classifiers',getattr(module,'attention_classifiers',None))
        for i,m in enumerate(members):
            x,y,w=module.datamodule.val_datasets[i].tensors
            with torch.no_grad():expected=(module.ce_loss(m(x),y)*w).sum()/w.sum()
            torch.testing.assert_close(trainer.callback_metrics[f'val_loss_stack_{i}'],expected,atol=1e-6,rtol=1e-6)
        self.rows.append([float(trainer.callback_metrics[f'val_loss_stack_{i}']) for i in range(len(members))])


def train_check(root,kind):
    train,val=datasets(kind)
    def run(tag,ids,stop=None,resume=None):
        directory=root/tag;directory.mkdir(parents=True,exist_ok=True)
        kw=dict(num_stacks=len(ids),num_classes=2,dim_quadjet_features=6,run_names=[str(i) for i in ids],stacked_run_name=tag,device=torch.device('cpu'))
        model=StackedFvTClassifier(**kw,dim_input_jet_features=4,dim_dijet_features=6,depth={'encoder':4,'decoder':1}) if kind=='fvt' else StackedAttentionClassifier(**kw,depth=1)
        members=model.fvt_classifiers if kind=='fvt' else model.attention_classifiers
        for pos,i in enumerate(ids):
            torch.manual_seed(500+i)
            ref=FvTClassifier(num_classes=2,dim_input_jet_features=4,dim_dijet_features=6,dim_quadjet_features=6,depth={'encoder':4,'decoder':1},run_name=str(i),device=torch.device('cpu')) if kind=='fvt' else AttentionClassifier(6,2,str(i),depth=1)
            members[pos].load_state_dict(ref.state_dict())
        model.optimizer_config={'type':'Adam','lr':.001}
        model.lr_scheduler_config={'type':'ReduceLROnPlateau','factor':.5,'patience':0,'threshold':1e-4,'cooldown':0,'min_lr':.0002}
        model.execution_chunk_size=0
        dm=IndependentStackedDataModule([train[i] for i in ids],[val[i] for i in ids],64,
             shuffle_seeds=[SEEDS[i] for i in ids],estimator_ids=[str(i) for i in ids],batch_size_milestones=[1,3])
        model.datamodule=dm;audit=Audit()
        saver=ResumableIndividualSaver(save_dir=directory/'models',run_names=[str(i) for i in ids],
              monitor_metrics=[f'val_loss_stack_{j}' for j in range(len(ids))],model='FvTClassifier' if kind=='fvt' else 'AttentionClassifier')
        callbacks=[saver,audit,ModelCheckpoint(dirpath=directory,save_last=True,save_top_k=0,save_on_train_epoch_end=True)]
        if stop is not None:callbacks.append(StopAfterEpoch(stop))
        trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=5,logger=False,callbacks=callbacks,
            reload_dataloaders_every_n_epochs=1,enable_progress_bar=False,enable_model_summary=False)
        trainer.fit(model,datamodule=dm,ckpt_path=str(resume) if resume else None)
        ck=torch.load(directory/'last.ckpt',map_location='cpu')
        return {i:copy.deepcopy(members[pos].state_dict()) for pos,i in enumerate(ids)},ck,audit.rows
    whole,ck,history=run('whole',[0,1,2]);permuted,_,_=run('permuted',[2,0,1])
    assert equal(whole,permuted),'Permutation changed sequential training'
    for i in range(3):
        solo,_,_=run(f'solo{i}',[i]);assert equal(whole[i],solo[i]),f'Solo model {i} changed'
    run('resumed',[0,1,2],stop=1)
    resumed,rck,_=run('resumed',[0,1,2],resume=root/'resumed/last.ckpt')
    assert equal(whole,resumed)
    assert equal(ck['optimizer_states'],rck['optimizer_states'])
    assert equal(ck['lr_schedulers'],rck['lr_schedulers'])
    counts=[next(iter(opt['state'].values()))['step'].item() for opt in ck['optimizer_states']]
    expected=[]
    for i in range(3):
        expected.append(sum(sum(b['members'][i] is not None for b in IndependentStackedDataModule(train,val,64,shuffle_seeds=SEEDS,batch_size_milestones=[1,3]).loader_for_epoch(epoch)) for epoch in range(5)))
    assert counts==expected,(counts,expected)
    return {'alone_and_permuted_exact':True,'resume_models_optimizers_schedulers_exact':True,
            'validation_all_rows_weighted_correctly':True,'optimizer_steps':counts,'expected_steps':expected}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('out',type=pathlib.Path);args=ap.parse_args()
    torch.set_num_threads(2);torch.autograd.set_detect_anomaly(False);args.out.mkdir(parents=True,exist_ok=True)
    result={'streams':stream_checks()}
    for kind in ('fvt','attention'):result[kind]=train_check(args.out/kind,kind)
    result['status']='PASS';(args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
