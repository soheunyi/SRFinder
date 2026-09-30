"""One artifact-backed training group, with completed-epoch recovery.

This worker does not submit jobs, choose scientific configurations, export
scores, or declare a campaign complete. Callers own process/GPU allocation.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import gc
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from constants import FEATURES
from independent_data import IndependentStackedDataModule, stream_identity
from member_initialization import initialize_members
from stacked_fvt import StackedFvTClassifier
from stacked_attention_classifier import StackedAttentionClassifier
from phase3.resumable import (AtomicCheckpointIO, ResumableIndividualSaver,
                              RngStateCallback, StopAfterEpoch)
from .training_store import canonical
from .member_splits import publish_member_splits, verify_member_splits
from .stage_completion import put_member_history,verify_member_history
from .memory_policy import choose_placement


def _atomic_json(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, mode='w', delete=False) as handle:
        tmp=Path(handle.name)
        try:
            json.dump(value,handle,indent=2,allow_nan=False)
            handle.write('\n');handle.flush();os.fsync(handle.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)


@contextmanager
def _owned_run(root, manifest, resume):
    existed=root.exists()
    if existed and not resume:raise ValueError('Output exists; explicit resume is required')
    root.mkdir(parents=True,exist_ok=resume)
    with (root/'.worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Training output already has an active worker') from exc
        path=root/'training-plan.json'
        if existed:
            if not path.is_file() or json.loads(path.read_text())!=manifest:
                raise ValueError('Training output has a different recipe or source version')
        else:_atomic_json(path,manifest)
        yield


def _implementation():
    root=Path(__file__).resolve().parents[1]
    names=('training_info.py','dataset.py','constants.py','smearing.py','utils.py',
           'independent_data.py','independent_training.py','member_initialization.py',
           'stacked_fvt.py','stacked_attention_classifier.py','fvt_classifier.py',
           'fvt_encoder.py','attention_classifier.py','network_blocks.py',
           'pl_callbacks.py','data_modules.py','phase3/resumable.py','phase2/identity.py',
           'artifacts/memory_policy.py','artifacts/stage_completion.py','artifacts/train_stage.py','artifacts/member_splits.py','artifacts/model_loading.py',
           'artifacts/source_context.py','artifacts/step1_context.py',
           'artifacts/step2_context.py','artifacts/step3_context.py','artifacts/regions.py')
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


class _IndexedSplit:
    def __init__(self,bank,indices):
        self.bank=bank
        self.device=bank[0].device
        self.indices=torch.as_tensor(indices,dtype=torch.long,device=self.device)

    def __len__(self):return len(self.indices)

    def gather(self,positions):
        selected=self.indices.index_select(0,positions)
        return tuple(t.index_select(0,selected) for t in self.bank)


class _TrainingHistory(pl.Callback):
    def __init__(self):self.records=[]

    def state_dict(self):return {'records':self.records}

    def load_state_dict(self,state):self.records=state['records']

    def on_validation_end(self,trainer,module):
        if trainer.sanity_checking:return
        epoch=int(trainer.current_epoch)
        if epoch!=len(self.records):raise ValueError('Nonconsecutive validation history')
        self.records.append({'epoch':epoch,'batch_size':module.datamodule.batch_size,
            'members':[{'val_loss':float(trainer.callback_metrics[f'val_loss_stack_{i}'].item()),
                        'lr':float(optimizer.param_groups[0]['lr'])}
                       for i,optimizer in enumerate(trainer.optimizers)]})

    def on_save_checkpoint(self,trainer,module,checkpoint):
        checkpoint['artifact_training_history']=module.history

    def on_load_checkpoint(self,trainer,module,checkpoint):
        module.history=checkpoint['artifact_training_history']


def _validate_tensors(tensors):
    x,y,w=tensors
    if x.dtype!=torch.float32 or w.dtype!=torch.float32 or y.dtype!=torch.long:
        raise ValueError('Expected float32 features/weights and integer labels')
    if not torch.isfinite(x).all() or not torch.isfinite(w).all() or not ((y==0)|(y==1)).all():
        raise ValueError('Invalid training features, labels or weights')


def _data_fingerprint(data,bank_hashes=None):
    if isinstance(data,_IndexedSplit):
        key=id(data.bank)
        if key not in bank_hashes:
            bank_hashes[key]=_data_fingerprint(TensorDataset(*data.bank))
        return {'bank':bank_hashes[key],
                'ordered_indices_sha256':hashlib.sha256(data.indices.cpu().numpy().tobytes()).hexdigest()}
    result=[]
    for tensor in data.tensors:
        tensor=tensor.detach().cpu().contiguous()
        result.append({'dtype':str(tensor.dtype),'shape':list(tensor.shape),
                       'sha256':hashlib.sha256(tensor.numpy().tobytes()).hexdigest()})
    return result


def train_stage(store, contexts, output, *, device='cpu', resident=False,
                resume=False, stop_after_completed_epochs=None,
                device_budget_bytes=None,compute_headroom_bytes=None):
    """Train configured full schedules, or stop at a recovery boundary for tests.

    Uses verified artifact contexts. Only best weights become permanent model
    artifacts; the rolling checkpoint and working files are retained until a
    later export/completion protocol validates the required outputs.
    """
    if not contexts:raise ValueError('At least one member is required')
    if resident is not False and resident is not True and resident!='auto':
        raise ValueError('resident must be False, True or auto')
    requested='auto' if resident=='auto' else ('resident' if resident else 'cpu')
    if device not in ('cpu','cuda') or (resident is True and device!='cuda'):
        raise ValueError('Residency requires an explicitly allocated CUDA device')
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    hps=[{k:v for k,v in c.hparams.items() if not k.startswith('aux_info')} for c in contexts]
    names=[c.hash for c in contexts]
    if len(set(names))!=len(names):raise ValueError('Duplicate member identities')
    hp=hps[0];stage=hp['step'];epochs=int(hp['max_epochs'])
    if stage not in (1,2,3) or epochs<1:raise ValueError('Unsupported stage or epoch count')
    shared=('step','model','max_epochs','depth','dim_quadjet_features','dim_dijet_features',
            'repr_norm','optimizer','lr_scheduler','dataloader')
    for other in hps[1:]:
        if any(other.get(key)!=hp.get(key) for key in shared):
            raise ValueError('One group requires matching architecture and training schedules')
    if hp['model']!=('AttentionClassifier' if stage==2 else 'FvTClassifier'):
        raise ValueError('Unsupported stage architecture')
    if stop_after_completed_epochs is not None and not 0<stop_after_completed_epochs<epochs:
        raise ValueError('Test stop must precede the configured final epoch')
    numerics={'device':device,'cpu_threads':torch.get_num_threads(),
              'matmul_precision':torch.get_float32_matmul_precision(),
              'cuda_matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
              'cudnn_tf32':torch.backends.cudnn.allow_tf32,
              'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
    if device=='cuda':
        numerics.update(gpu_name=torch.cuda.get_device_name(),
                        compute_capability=list(torch.cuda.get_device_capability()))
    manifest={'schema':1,'members':names,'hparams':hps,'source_sha256':_implementation(),
              'torch_version':torch.__version__,'lightning_version':pl.__version__,
              'numerics':numerics,'dtype':'float32','alignment':32,'retain_validation':True,
              'resume_boundary':'completed_epoch'}
    root=Path(output)
    with _owned_run(root,manifest,resume):
        marker=root/'training-completion.json'
        if marker.exists():
            result=json.loads(marker.read_text())
            if result.get('status')!='TRAINING_COMPLETE_EXPORT_PENDING' or result.get('completed_epochs')!=epochs:
                raise ValueError('Invalid training-completion record')
            if len(result['model_ids'])!=len(contexts):raise ValueError('Incomplete model set')
            if len(result['history_ids'])!=len(contexts):raise ValueError('Incomplete member histories')
            for key,hkey,context in zip(result['model_ids'],result['history_ids'],contexts):
                verify_member_history(store,hkey,key)
                record=store.read(key,'model');store.payload_path(key)
                if record['identity']['estimator']!={'stage':stage,'context_identity':context.hash}:
                    raise ValueError('Completed model identity differs')
                recipe=record['identity']['training_recipe']
                expected_hp={k:v for k,v in context.hparams.items() if not k.startswith('aux_info')}
                if (recipe['hparams']!=expected_hp or recipe['implementation_sha256']!=manifest['source_sha256']
                        or recipe['numerics']!=numerics):
                    raise ValueError('Completed model recipe differs')
                verify_member_splits(store,context,record['identity']['split_ids'])
            return result
        begin=time.perf_counter()
        pairs=[];splits=[];banks={}
        for context in contexts:
            split=publish_member_splits(store,context)
            rows=verify_member_splits(store,context,split)
            if stage==2:
                pair=context.fetch_train_val_smeared_features(FEATURES,'fourTag','weight',
                    device=device,training_alignment=32,retain_validation=True)
            else:
                selected=np.flatnonzero(context.ms_idx).astype(np.int64)
                bank_key=(context.hparams['source_dataset_id'],hashlib.sha256(selected.tobytes()).hexdigest())
                if bank_key not in banks:
                    frame=context.scdinfo.fetch_data()
                    if len(frame)!=len(selected):raise ValueError('Source bank row count differs')
                    bank=(torch.tensor(frame[FEATURES].values,dtype=torch.float32),
                          torch.tensor(frame.fourTag.values,dtype=torch.long),
                          torch.tensor(frame.weight.values,dtype=torch.float32))
                    _validate_tensors(bank)
                    banks[bank_key]=bank
                bank=banks[bank_key]
                mapped=[np.searchsorted(selected,indices) for indices in rows]
                if any(not np.array_equal(selected[local],indices) for local,indices in zip(mapped,rows)):
                    raise ValueError('Split rows lie outside the shared bank')
                pair=tuple(_IndexedSplit(bank,indices) for indices in mapped)
            for indices,data in zip(rows,pair):
                if len(indices)!=len(data):raise ValueError('Reconstructed row count differs')
                if not isinstance(data,_IndexedSplit):_validate_tensors(data.tensors)
            pairs.append(pair);splits.append(split)
        bank_hashes={}
        fingerprint=[[_data_fingerprint(data,bank_hashes) for data in pair] for pair in pairs]
        bank_bytes=sum(t.numel()*t.element_size() for bank in banks.values() for t in bank)
        data_bytes=(bank_bytes+sum(d.indices.numel()*d.indices.element_size() for pair in pairs for d in pair)
                    if stage!=2 else sum(t.numel()*t.element_size() for pair in pairs for d in pair for t in d.tensors))
        available=None
        if device=='cuda':
            gc.collect();torch.cuda.empty_cache()
            available=int(torch.cuda.mem_get_info()[0])
        memory=choose_placement(requested,device=device,data_bytes=data_bytes,budget_bytes=device_budget_bytes,
                                headroom_bytes=compute_headroom_bytes,available_bytes=available)
        _atomic_json(root/'memory-decision.json',memory)
        resident=memory['placement']=='resident'
        if resident and stage!=2:
            placed={id(bank):tuple(t.to('cuda') for t in bank) for bank in banks.values()}
            pairs=[tuple(_IndexedSplit(placed[id(data.bank)],data.indices) for data in pair) for pair in pairs]
        receipt=root/'training-inputs.json'
        if receipt.exists():
            if json.loads(receipt.read_text())!=fingerprint:raise ValueError('Training tensors changed on resume')
        else:_atomic_json(receipt,fingerprint)
        kwargs=dict(num_stacks=len(contexts),num_classes=2,dim_quadjet_features=hp['dim_quadjet_features'],
                    run_names=names,stacked_run_name='artifact-stage',device='cpu',depth=hp['depth'])
        if stage==2:
            model=StackedAttentionClassifier(**kwargs);members=model.attention_classifiers
        else:
            model=StackedFvTClassifier(**kwargs,dim_input_jet_features=4,
                dim_dijet_features=hp['dim_dijet_features'],repr_norm=hp.get('repr_norm',False))
            members=model.fvt_classifiers
        policies=initialize_members(members,hps)
        model.optimizer_config=hp['optimizer'];model.lr_scheduler_config=hp['lr_scheduler']
        model.execution_chunk_size=0
        dc=hp['dataloader']
        dm=IndependentStackedDataModule([p[0] for p in pairs],[p[1] for p in pairs],dc['batch_size'],
            shuffle_seeds=[h['train_seed'] for h in hps],estimator_ids=[stream_identity(h) for h in hps],
            batch_size_milestones=dc.get('batch_size_milestones',[]),
            batch_size_multiplier=dc.get('batch_size_multiplier',2),num_workers=0,
            storage_device='cuda' if resident and stage==2 else None)
        model.datamodule=dm
        saver=ResumableIndividualSaver(save_dir=root/'models',run_names=names,
            monitor_metrics=[f'val_loss_stack_{i}' for i in range(len(names))],model=hp['model'])
        history=_TrainingHistory()
        callbacks=[saver,RngStateCallback(),history,ModelCheckpoint(dirpath=root,save_last=True,
            save_top_k=0,save_on_train_epoch_end=True)]
        if stop_after_completed_epochs is not None:
            callbacks.append(StopAfterEpoch(stop_after_completed_epochs-1))
        trainer=pl.Trainer(accelerator='gpu' if device=='cuda' else 'cpu',devices=1,precision='32-true',
            max_epochs=epochs,logger=False,callbacks=callbacks,plugins=[AtomicCheckpointIO()],
            num_sanity_val_steps=0,reload_dataloaders_every_n_epochs=1,
            enable_progress_bar=False,enable_model_summary=False)
        checkpoint=root/'last.ckpt'
        fit_start=time.perf_counter()
        trainer.fit(model,datamodule=dm,ckpt_path=str(checkpoint) if resume and checkpoint.exists() else None)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        completed=int(state['epoch'])+1
        result={'status':'INTERRUPTED_RECOVERABLE','completed_epochs':completed,
                'shared_raw_banks':len(banks),'raw_bank_bytes':bank_bytes,'memory_policy':memory,
                'preparation_s':fit_start-begin,'fit_checkpoint_s':time.perf_counter()-fit_start}
        if completed!=epochs:return result
        ids=[]
        for i,(context,split,policy) in enumerate(zip(contexts,splits,policies)):
            metric=f'val_loss_stack_{i}'
            if saver.best_epochs[metric]<0:raise ValueError('No finite best checkpoint was selected')
            ids.append(store.put_model(root/'models'/f'{context.hash}_best.pt',
                {'stage':stage,'context_identity':context.hash},
                {'hparams':hps[i],'initialization_policy':policy,
                 'implementation_sha256':manifest['source_sha256'],'numerics':numerics,
                 'best_epoch':saver.best_epochs[metric],'best_val_loss':saver.best_scores[metric]},split))
        histories=[put_member_history(store,key,[{'epoch':r['epoch'],'batch_size':r['batch_size'],
                   **r['members'][i]} for r in history.records]) for i,key in enumerate(ids)]
        result.update(status='TRAINING_COMPLETE_EXPORT_PENDING',model_ids=ids,history_ids=histories)
        _atomic_json(marker,result)
        return result
