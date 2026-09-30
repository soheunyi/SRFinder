"""Real Lightning hook-order regression for per-estimator checkpoint selection."""
import argparse,json,pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE.parent,HERE.parent/'phase1',HERE.parent/'phase3'):
    sys.path.insert(0,str(p))
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_callbacks import SaveIndividualClassifierCallback
from resumable import ResumableIndividualSaver,StopAfterEpoch
from fingerprint import FingerprintCallback

SCORES=((3.,1.,2.,4.),(.5,.5,2.,3.))
class Probe(pl.LightningModule):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        setattr(self,'fvt_classifiers' if kind=='FvTClassifier' else 'attention_classifiers',
                nn.ModuleList([nn.Linear(1,1,bias=False) for _ in range(2)]))
        for p in self.parameters():p.data.fill_(-777.)
    @property
    def members(self):return getattr(self,'fvt_classifiers' if self.kind=='FvTClassifier' else 'attention_classifiers')
    def on_train_epoch_start(self):
        with torch.no_grad():
            for i,m in enumerate(self.members):m.weight.fill_(100+10*self.current_epoch+i)
    def training_step(self,batch,batch_idx):return sum(p.sum()*0 for p in self.parameters())
    def validation_step(self,batch,batch_idx):pass
    def on_validation_epoch_end(self):
        for i in range(2):
            score=-1000. if self.trainer.sanity_checking else SCORES[i][self.current_epoch]
            self.log(f'val_loss_stack_{i}',score,on_epoch=True)
    def configure_optimizers(self):return torch.optim.SGD(self.parameters(),lr=0.)
    def train_dataloader(self):return DataLoader(TensorDataset(torch.zeros(1,1),torch.zeros(1,1),torch.ones(1,1)),batch_size=1)
    def val_dataloader(self):return self.train_dataloader()

class VerifyEachEpoch(pl.Callback):
    def __init__(self,saver,directory,resuming=False):super().__init__();self.saver=saver;self.directory=directory;self.resuming=resuming;self.epochs=[];self.sanity=0
    def on_validation_end(self,trainer,module):
        if trainer.sanity_checking:
            self.sanity+=1
            if not self.resuming:
                assert not list(self.directory.glob('*.pt')),'Sanity validation wrote checkpoints'
                assert all(v==float('inf') for v in self.saver.best_scores.values())
            return
        epoch=trainer.current_epoch
        for i in range(2):
            best_epoch=min(range(epoch+1),key=lambda e:SCORES[i][e])
            best=torch.load(self.directory/f'm{i}_best.pt',map_location='cpu')
            last=torch.load(self.directory/f'm{i}_last.pt',map_location='cpu')
            assert best['weight'].item()==100+10*best_epoch+i,(epoch,i,best)
            assert last['weight'].item()==100+10*epoch+i
            assert self.saver.best_scores[f'val_loss_stack_{i}']==SCORES[i][best_epoch]
        self.epochs.append(epoch)

def run(root,kind,callback_type,stop=None,resume=None):
    models=root/'models';models.mkdir(parents=True,exist_ok=True)
    saver=callback_type(save_dir=str(models),run_names=['m0','m1'],monitor_metrics=['val_loss_stack_0','val_loss_stack_1'],model=kind)
    check=VerifyEachEpoch(saver,models,resuming=resume is not None)
    callbacks=[saver,check]
    fingerprint=None
    if kind=='FvTClassifier':
        fingerprint=FingerprintCallback(2,root/('fingerprint_resume.json' if resume else 'fingerprint.json'))
        callbacks.append(fingerprint)
    if stop is not None:callbacks.append(StopAfterEpoch(stop))
    callbacks.append(ModelCheckpoint(dirpath=root,save_last=True,save_top_k=0,every_n_epochs=1,save_on_train_epoch_end=True))
    trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=4,logger=False,callbacks=callbacks,
       enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=1)
    trainer.fit(Probe(kind),ckpt_path=str(resume) if resume else None)
    if fingerprint:
        rec=fingerprint.record
        # Resumable saver metadata includes best epochs before interruption.
        if resume is not None or stop is None:
            assert rec['best']['epochs']==[1,0]
            assert rec['best']['scores']==[1.,.5]
        assert rec['saver_best_scores']==[1.,.5]
        assert all(v['val_loss_per_stack_as_seen_by_saver']==v['val_loss_per_stack'] for v in rec['val_epochs'])
    assert check.sanity>0 or resume is not None
    return torch.load(root/'last.ckpt',map_location='cpu'),check.epochs

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    ap=argparse.ArgumentParser();ap.add_argument('out',type=pathlib.Path);args=ap.parse_args()
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True);report={}
    for kind in ('FvTClassifier','AttentionClassifier'):
        for cls in (SaveIndividualClassifierCallback,ResumableIndividualSaver):
            key=f'{kind}_{cls.__name__}'
            whole,epochs=run(args.out/key/'whole',kind,cls)
            result={'epochs_checked':epochs,'sanity_excluded':True,'epoch_zero_saved':True,'best_epoch_weights_correct':True}
            if cls is ResumableIndividualSaver:
                split=args.out/key/'split'
                run(split,kind,cls,stop=1)
                resumed,_=run(split,kind,cls,resume=split/'last.ckpt')
                assert equal(whole['state_dict'],resumed['state_dict'])
                assert equal(whole['optimizer_states'],resumed['optimizer_states'])
                for i in range(2):
                    assert equal(torch.load(args.out/key/'whole/models'/f'm{i}_best.pt'),torch.load(split/'models'/f'm{i}_best.pt'))
                key_state=next(k for k,v in whole['callbacks'].items() if isinstance(v,dict) and 'best_epochs' in v)
                assert equal(whole['callbacks'][key_state],resumed['callbacks'][key_state])
                result['resume_preserves_best_and_state']=True
            report[key]=result
    (args.out/'result.json').write_text(json.dumps({'status':'PASS','checks':report},indent=2)+'\n')
    print(json.dumps({'status':'PASS','checks':report},indent=2),flush=True)

if __name__=='__main__':main()
