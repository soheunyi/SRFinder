"""Small CPU workflow smoke: new weights/scores flow through all three stages.

Uses synthetic overlapping classes and two members/two epochs per stage.
Unique binary-exact weights identify fixture rows for split-receipt checks.
This is not a scientific experiment or a full-campaign launcher.
"""
import argparse
import copy
import hashlib
import pathlib
import sys
import json
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'phase3')]
from constants import FEATURES
from dataset import SCDatasetInfo
from independent_data import IndependentStackedDataModule,stream_identity
from member_initialization import initialize_members
from stacked_fvt import StackedFvTClassifier
from stacked_attention_classifier import StackedAttentionClassifier
from resumable import ResumableIndividualSaver,AtomicCheckpointIO
from artifacts.training_store import TrainingStore,canonical
from artifacts.member_splits import publish_member_splits,verify_member_splits
from artifacts.source_context import verify_source_context
from artifacts.step1_context import ArtifactStep1Context
from artifacts.step2_context import ArtifactStep2Context
from artifacts.step3_context import ArtifactStep3Context
from artifacts.model_loading import load_model
from artifacts.export_scores import export_member_scores
from artifacts.regions import define_regions,classify_X2
from artifacts.affine_inputs import prepare_affine_inputs

N=2048


def hparams(stage,member):
    return {'experiment_name':f'artifact_workflow_smoke_step{stage}','step':stage,
            'model':'AttentionClassifier' if stage==2 else 'FvTClassifier',
            'model_seed':member,'train_seed':member,'data_seed':member,'ensemble_member':member,
            'val_ratio':.33,'fit_batch_size':1024,'max_epochs':2,'repr_norm':False,
            'dim_dijet_features':6,'dim_quadjet_features':6,
            'depth':8 if stage==2 else {'encoder':4,'decoder':1},
            'optimizer':{'type':'Adam','lr':.01},
            'lr_scheduler':{'type':'ReduceLROnPlateau','factor':.25 if stage==2 else .5,
                            'patience':3 if stage==2 else 10,'threshold':1e-4,'cooldown':1,'min_lr':.0002},
            'dataloader':{'batch_size':64,'batch_size_multiplier':2,'batch_size_milestones':[1]},
            'smearing':{'noise_scale':2.,'seed':0,'hard_cutoff':False,'scale_mode':'std'}}


def train(store,source,contexts,directory):
    directory.mkdir(parents=True)
    stage=contexts[0].hparams['step'];hp=contexts[0].hparams
    names=[f'member-{i}' for i in range(len(contexts))]
    datasets=[]
    for context in contexts:
        if stage==2:
            pair=context.fetch_train_val_smeared_features(FEATURES,'fourTag','weight',device='cpu',training_alignment=32,retain_validation=True)
        else:
            pair=context.fetch_train_val_tensor_datasets(FEATURES,'fourTag','weight',training_alignment=32,retain_validation=True)
        datasets.append(pair)
    kwargs=dict(num_stacks=len(contexts),num_classes=2,dim_quadjet_features=6,
                run_names=names,stacked_run_name=f'step-{stage}',device='cpu',depth=hp['depth'])
    if stage==2:model=StackedAttentionClassifier(**kwargs)
    else:model=StackedFvTClassifier(**kwargs,dim_input_jet_features=4,dim_dijet_features=6,repr_norm=False)
    members=model.attention_classifiers if stage==2 else model.fvt_classifiers
    policies=initialize_members(members,[c.hparams for c in contexts])
    model.optimizer_config=hp['optimizer'];model.lr_scheduler_config=hp['lr_scheduler'];model.execution_chunk_size=0
    dm=IndependentStackedDataModule([d[0] for d in datasets],[d[1] for d in datasets],64,
        shuffle_seeds=[c.hparams['train_seed'] for c in contexts],
        estimator_ids=[stream_identity(c.hparams) for c in contexts],batch_size_milestones=[1])
    model.datamodule=dm
    saver=ResumableIndividualSaver(save_dir=directory/'models',run_names=names,
        monitor_metrics=[f'val_loss_stack_{i}' for i in range(len(contexts))],model=hp['model'])
    trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=2,logger=False,
        callbacks=[saver,ModelCheckpoint(dirpath=directory,save_last=True,save_top_k=0,save_on_train_epoch_end=True)],
        plugins=[AtomicCheckpointIO()],num_sanity_val_steps=0,reload_dataloaders_every_n_epochs=1,
        enable_progress_bar=False,enable_model_summary=False)
    trainer.fit(model,datamodule=dm)
    ids=[]
    for m,context in enumerate(contexts):
        rng_before=np.random.get_state()
        split_ids=publish_member_splits(store,context)
        rng_after=np.random.get_state()
        assert rng_before[0]==rng_after[0] and rng_before[2:]==rng_after[2:]
        np.testing.assert_array_equal(rng_before[1],rng_after[1])
        reconstructed=verify_member_splits(store,context,split_ids)
        with_aux=copy.copy(context)
        with_aux._aux_info={'historical_score_array':np.arange(5)}
        assert publish_member_splits(store,with_aux)==split_ids
        # Independent fixture oracle: actual tensor weights encode source rows.
        for rows,data in zip(reconstructed,datasets[m]):
            observed=np.rint((data.tensors[2].numpy()-1)*N).astype(np.int64)
            np.testing.assert_array_equal(rows,observed)
            assert len(np.unique(rows))==len(rows)
            assert np.all(context.ms_idx[rows])
        try:
            verify_member_splits(store,context,split_ids[::-1])
        except ValueError:
            pass
        else:
            raise AssertionError('Swapped split receipts accepted')
        metric=f'val_loss_stack_{m}'
        ids.append(store.put_model(directory/'models'/f'{names[m]}_best.pt',
            {'stage':stage,'member':m,'context_identity':context.hash},
            {'hparams':context.hparams,'initialization_policy':policies[m],
             'best_epoch':saver.best_epochs[metric],'best_val_loss':saver.best_scores[metric]},split_ids))
    return ids


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=pathlib.Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(71)
    torch.autograd.set_detect_anomaly(False)
    x=torch.rand(N,4,4)
    x[:,0]=40+100*x[:,0];x[:,1]=2*x[:,1]-1;x[:,2]=6*x[:,2]-3;x[:,3]=5+15*x[:,3]
    frame=pd.DataFrame(x.reshape(N,16).numpy(),columns=FEATURES)
    rng=np.random.RandomState(19)
    frame['fourTag']=rng.binomial(1,.25+.5/(1+np.exp(-(frame[FEATURES[:4]].mean(axis=1)-90)/20)))
    frame['weight']=1+np.arange(N,dtype=np.float64)/N
    paths=[args.out/'synthetic_pool_0.h5',args.out/'synthetic_pool_1.h5']
    parts=[frame.iloc[:733],frame.iloc[733:]]
    for path,part in zip(paths,parts):part.to_hdf(path,key='df')
    raw=SCDatasetInfo([p.resolve() for p in paths],[np.ones(len(part),dtype=bool) for part in parts])
    params={'seed':7,'n_3b':int((frame.fourTag==0).sum()),'ratio_4b':float(frame.loc[frame.fourTag==1,'weight'].sum()/frame.weight.sum()),
            'signal_ratio':0.,'signal_filename':paths[1].name,'base_fvt_train_ratio':.5}
    source_hp={**hparams(1,0),'dataset':params}
    desc={'pools':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
          'mother_parameters':params,'mother_selection_sha256':[hashlib.sha256(mask.tobytes()).hexdigest() for mask in raw.inner_idxs]}
    store=TrainingStore(args.out/'store');dataset=store.put_dataset(hashlib.sha256(canonical(desc)).hexdigest(),N,desc)
    source=verify_source_context(store,dataset,source_hp,raw,source_root=args.out)
    splits={d:store.put_split(dataset,d,source.indices(d),{'algorithm':'outer_seed_shuffle','version':1,'seed':7,'fraction':.5}) for d in ('X1','X2')}
    event_ids={}
    for d in ('X1','X2'):
        rows=source.indices(d);a=np.empty(len(rows),dtype=[('is_4b','?'),('weight','<f8')])
        a['is_4b']=frame.fourTag.values[rows];a['weight']=frame.weight.values[rows]
        event_ids[d]=store.put_array('event_metadata',{'dataset_id':dataset,'split_id':splits[d]},a)
    def export(models,domains,encoders=None):
        result={d:[] for d in domains}
        for m,owner in enumerate(models):
            encoder=load_model(store,encoders[m]) if encoders is not None else None
            for d in domains:
                rows=source.indices(d)
                features=torch.tensor(frame.iloc[rows][FEATURES].values,dtype=torch.float32)
                if encoder is not None:
                    with torch.inference_mode():features=encoder.encoder(features)
                batches=[(rows[i:i+128],features[i:i+128]) for i in range(0,len(rows),128)]
                result[d].append(export_member_scores(store,owner,splits[d],rows,batches))
        return result
    base=train(store,source,[ArtifactStep1Context(source,hparams(1,m)) for m in range(2)],args.out/'step1')
    base_scores=export(base,('X1','X2'))
    smooth=train(store,source,[ArtifactStep2Context(store,base[m],source,dataset,hparams(2,m)) for m in range(2)],args.out/'step2')
    smooth_scores=export(smooth,('X1','X2'),encoders=base)
    region=define_regions(store,base_scores['X1'],event_ids['X1'],smooth_scores['X1'],sr_fraction=.2,cr_fraction=.8)
    cr=train(store,source,[ArtifactStep3Context(store,region,source,base_scores['X2'],smooth_scores['X2'],hparams(3,m)) for m in range(2)],args.out/'step3')
    cr_scores=export(cr,('X2',))
    ensemble=store.put_ensemble(cr,'mean_probability')
    before=store.storage_stats();log_ratio=store.aggregate_scores(ensemble,cr_scores['X2']);assert store.storage_stats()==before
    groups=classify_X2(store,region,base_scores['X2'],smooth_scores['X2'])
    rows=source.indices('X2');labels=frame.fourTag.values[rows];weights=frame.weight.values[rows]
    inputs,audit=prepare_affine_inputs(np.column_stack([np.zeros(len(rows),dtype=np.int64),rows]),labels,weights,
        groups['log_psi'],log_ratio,dataset_id=dataset,domain='X2',lower=store.read(region)['payload']['log_tau_s'])
    assert audit['positive_normalizers'] and all(np.isfinite(v).all() for v in inputs)
    assert len(cr_scores['X2'])==2 and all(len(store.load_array(key))==len(rows) for key in cr_scores['X2'])
    result={'status':'PASS','scope':'synthetic CPU integration, two members and two epochs per stage; no bootstrap inference',
            'models':{'step1':base,'step2':smooth,'step3':cr},'region':region,'all_X2_rows':len(rows),
            'test_input_counts':{'3b':audit['n3'],'4b':audit['n4']},'storage':store.storage_stats()}
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','stages':3,'members_per_stage':2,'all_X2_rows':len(rows)}),flush=True)


if __name__=='__main__':main()
