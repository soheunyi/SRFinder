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
import pickle
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'phase3')]
from constants import FEATURES
from dataset import SCDatasetInfo,MotherSamples
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
from artifacts.export_stage import export_stage
from artifacts.regions import define_regions,classify_X2
from artifacts.affine_inputs import prepare_affine_inputs
from artifacts.train_stage import train_stage, _TrainingHistory
from artifacts.stage_completion import complete_stage,verify_stage_completion
from artifacts.run_stage import run_stage
from artifacts.bound_tasks import register_source_pointer,make_stage_task,build_stage_contexts
from artifacts.materialize_task import materialize_task
from artifacts.case_registry import CaseRegistry

N=2048
TEST_DEVICE="cpu"
TRAIN_OPTIONS={}


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
            'smearing':{'noise_scale':2.,'seed':7 if stage==2 else 0,'hard_cutoff':False,'scale_mode':'std'}}


def reference_train(store,source,contexts,directory):
    directory.mkdir(parents=True)
    stage=contexts[0].hparams['step'];hp=contexts[0].hparams
    names=[f'member-{i}' for i in range(len(contexts))]
    datasets=[]
    for context in contexts:
        if stage==2:
            pair=context.fetch_train_val_smeared_features(FEATURES,'fourTag','weight',device=TEST_DEVICE,training_alignment=32,retain_validation=True)
        elif context.hparams['model']=='AttentionClassifier':
            pair=context.fetch_train_val_representation_datasets(FEATURES,'fourTag','weight',device=TEST_DEVICE,training_alignment=32,retain_validation=True)
        else:
            pair=context.fetch_train_val_tensor_datasets(FEATURES,'fourTag','weight',training_alignment=32,retain_validation=True)
        datasets.append(pair)
    kwargs=dict(num_stacks=len(contexts),num_classes=2,dim_quadjet_features=6,
                run_names=names,stacked_run_name=f'step-{stage}',device='cpu',depth=hp['depth'])
    if hp['model']=='AttentionClassifier':model=StackedAttentionClassifier(**kwargs)
    else:model=StackedFvTClassifier(**kwargs,dim_input_jet_features=4,dim_dijet_features=6,repr_norm=False)
    members=model.attention_classifiers if hp['model']=='AttentionClassifier' else model.fvt_classifiers
    policies=initialize_members(members,[c.hparams for c in contexts])
    model.optimizer_config=hp['optimizer'];model.lr_scheduler_config=hp['lr_scheduler'];model.execution_chunk_size=0
    dm=IndependentStackedDataModule([d[0] for d in datasets],[d[1] for d in datasets],64,
        shuffle_seeds=[c.hparams['train_seed'] for c in contexts],
        estimator_ids=[stream_identity(c.hparams) for c in contexts],batch_size_milestones=[1])
    model.datamodule=dm
    saver=ResumableIndividualSaver(save_dir=directory/'models',run_names=names,
        monitor_metrics=[f'val_loss_stack_{i}' for i in range(len(contexts))],model=hp['model'])
    trainer=pl.Trainer(accelerator='gpu' if TEST_DEVICE=='cuda' else 'cpu',devices=1,max_epochs=2,logger=False,
        callbacks=[saver,_TrainingHistory(),ModelCheckpoint(dirpath=directory,save_last=True,save_top_k=0,save_on_train_epoch_end=True)],
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


def assert_tree_equal(left,right):
    if isinstance(left,torch.Tensor):torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:assert_tree_equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):assert_tree_equal(a,b)
    else:assert left==right


def train(store,source,contexts,directory):
    reference=reference_train(store,source,contexts,directory/'reference')
    output=directory/'worker'
    prefix=train_stage(store,contexts,output,**TRAIN_OPTIONS,resident='auto',stop_after_completed_epochs=1)
    assert prefix['memory_policy']['placement']==('resident' if TEST_DEVICE=='cuda' else 'cpu')
    assert prefix['status']=='INTERRUPTED_RECOVERABLE' and prefix['completed_epochs']==1
    assert not (output/'training-completion.json').exists()
    result=train_stage(store,contexts,output,**TRAIN_OPTIONS,resume=True)
    assert result['status']=='TRAINING_COMPLETE_EXPORT_PENDING' and result['completed_epochs']==2
    assert result['shared_raw_banks']==(0 if contexts[0].hparams['model']=='AttentionClassifier' else 1)
    for expected,actual in zip(reference,result['model_ids']):
        assert_tree_equal(load_model(store,expected).state_dict(),load_model(store,actual).state_dict())
        expected_recipe=store.read(expected,'model')['identity']['training_recipe']
        actual_recipe=store.read(actual,'model')['identity']['training_recipe']
        for key in ('best_epoch','best_val_loss'):
            assert expected_recipe[key]==actual_recipe[key]
    ref=torch.load(directory/'reference'/'last.ckpt',map_location='cpu',weights_only=False)
    actual=torch.load(output/'last.ckpt',map_location='cpu',weights_only=False)
    for name in ('state_dict','optimizer_states','lr_schedulers','artifact_training_history'):
        assert_tree_equal(ref[name],actual[name])
    with patch.object(pl.Trainer,'fit',side_effect=AssertionError('Completed training repeated')):
        assert train_stage(store,contexts,output,**TRAIN_OPTIONS,resume=True)==result
    changed=copy.copy(contexts[0]);changed._hparams=copy.deepcopy(contexts[0]._hparams)
    changed._hparams['train_seed']+=1
    try:train_stage(store,[changed,*contexts[1:]],output,**TRAIN_OPTIONS,resume=True)
    except ValueError:pass
    else:raise AssertionError('Changed training recipe accepted on resume')
    coordinated=directory/'coordinated'
    prefix=run_stage(store,contexts,source,coordinated,**TRAIN_OPTIONS,resident='auto',export_batch_size=128,stop_after_completed_epochs=1)
    assert prefix['status']=='INTERRUPTED_RECOVERABLE'
    with patch('artifacts.run_stage.export_stage',side_effect=RuntimeError('Synthetic export interruption')):
        try:run_stage(store,contexts,source,coordinated,**TRAIN_OPTIONS,resume=True,export_batch_size=128)
        except RuntimeError as exc:assert str(exc)=='Synthetic export interruption'
        else:raise AssertionError('Expected interrupted export')
    assert not (coordinated/'stage-completion.json').exists()
    with patch.object(pl.Trainer,'fit',side_effect=AssertionError('Export retry retrained models')):
        done=run_stage(store,contexts,source,coordinated,**TRAIN_OPTIONS,resume=True,export_batch_size=128)
        assert run_stage(store,contexts,source,coordinated,**TRAIN_OPTIONS,resume=True,export_batch_size=128)==done
    assert done['status']=='STAGE_ARTIFACTS_COMPLETE'
    for expected,actual in zip(result['model_ids'],done['model_ids']):
        assert_tree_equal(load_model(store,expected).state_dict(),load_model(store,actual).state_dict())
    return result['model_ids']


def main():
    global TEST_DEVICE,TRAIN_OPTIONS
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=ap.parse_args()
    TEST_DEVICE=args.device
    if TEST_DEVICE=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required for this check')
    if TEST_DEVICE=='cuda':
        torch.set_float32_matmul_precision('medium')
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=False
    TRAIN_OPTIONS={'device':TEST_DEVICE,'device_budget_bytes':2*1024**3,'compute_headroom_bytes':1024**3}
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
    mother_path=args.out/'shared-mother.pkl'
    mother_path.write_bytes(pickle.dumps(MotherSamples(raw,'fixture',params)))
    pointer=register_source_pointer(store,mother_path,source_hp,source_root=args.out)
    assert pointer['dataset_id']==dataset
    case_receipts={};case_tasks={}
    def logical_node(stage,*,seeds=None,sr=.2,model=None):
        seeds=[0,1] if seeds is None else seeds
        return {'id':f'fixture-step{stage}','stage':stage,'member_count':len(seeds),'member_seeds':seeds,
                'max_epochs':2,'axes':{'mother_seed':7,'epsilon':'0','eta':'2.0','sr_fraction':str(sr),
                                     'model':model or ('AttentionClassifier' if stage==2 else 'FvTClassifier')},
                'requires':[] if stage==1 else ['fixture-step1'] if stage==2 else ['fixture-step1','fixture-step2'],
                'region_member_seeds':seeds,
                **({'region_recipe':'sr_quantile_cr_complement_v2'} if stage==3 else {})}
    def bound(stage,upstream=None):
        node=logical_node(stage)
        task=materialize_task(store,node,pointer,[hparams(stage,m) for m in range(2)],
            {key:case_receipts[key] for key in node['requires']},expected_dataset=params)
        assert task['upstream']==(upstream or {})
        case_tasks[node['id']]=task
        contexts,reconstructed=build_stage_contexts(store,task)
        np.testing.assert_array_equal(reconstructed.ms_idx,source.ms_idx)
        assert all(c.hparams['source_dataset_id']==dataset for c in contexts)
        return contexts
    registry_plan={'schema':1,'nodes':[logical_node(stage) for stage in (1,2,3)]}
    registry=CaseRegistry(store,registry_plan,args.out/'case-registry')
    assert registry.parents('fixture-step2') is None
    changed={**pointer,'mother_record_sha256':'0'*64}
    try:build_stage_contexts(store,make_stage_task(changed,[hparams(1,0)]))
    except ValueError:pass
    else:raise AssertionError('Changed mother record accepted')
    splits={d:store.put_split(dataset,d,source.indices(d),{'algorithm':'outer_seed_shuffle','version':1,'seed':7,'fraction':.5}) for d in ('X1','X2')}
    event_ids={}
    for d in ('X1','X2'):
        rows=source.indices(d);a=np.empty(len(rows),dtype=[('is_4b','?'),('weight','<f8')])
        a['is_4b']=frame.fourTag.values[rows];a['weight']=frame.weight.values[rows]
        event_ids[d]=store.put_array('event_metadata',{'dataset_id':dataset,'split_id':splits[d]},a)
    def export(models,domains,encoders=None):
        result=export_stage(store,models,source,device=TEST_DEVICE,batch_size=128)
        assert set(result['score_ids'])==set(domains)
        splits.update(result['evaluation_splits']);event_ids.update(result['event_metadata_ids'])
        # Independent direct evaluation in the same fixed inference batches.
        for m,owner in enumerate(models):
            model=load_model(store,owner,device=TEST_DEVICE)
            encoder=load_model(store,encoders[m],device=TEST_DEVICE) if encoders is not None else None
            for domain in domains:
                rows=source.indices(domain);predictions=[]
                with torch.inference_mode():
                    for begin in range(0,len(rows),128):
                        x=torch.tensor(frame.iloc[rows[begin:begin+128]][FEATURES].values,dtype=torch.float32).contiguous().to(TEST_DEVICE)
                        if encoder is not None:x=encoder.encoder(x).contiguous()
                        logits=model(x);predictions.append((logits[:,1]-logits[:,0]).cpu().numpy())
                np.testing.assert_array_equal(store.load_array(result['score_ids'][domain][m]),np.concatenate(predictions))
                metadata=store.load_array(event_ids[domain])
                assert len(metadata)==len(rows)
                np.testing.assert_array_equal(metadata['weight'],frame.weight.values[rows])
                np.testing.assert_array_equal(metadata['pool'],(rows>=733).astype(np.uint32))
                np.testing.assert_array_equal(metadata['pool_row'],np.where(rows<733,rows,rows-733))
        with patch.object(SCDatasetInfo,'fetch_data',side_effect=AssertionError('Cached export read raw features')), \
             patch('artifacts.export_stage.export_member_scores',side_effect=AssertionError('Cached export ran inference')):
            assert export_stage(store,models,source,device=TEST_DEVICE,batch_size=128)==result
        return result['score_ids']
    completions=[]
    def finish(stage,scores,name=None):
        result=json.loads((args.out/(name or f'step{stage}')/'worker'/'training-completion.json').read_text())
        domain_splits={domain:splits[domain] for domain in scores}
        key=complete_stage(store,result,source,domain_splits,scores,{d:event_ids[d] for d in scores})
        assert verify_stage_completion(store,key,source)
        assert complete_stage(store,result,source,domain_splits,scores,{d:event_ids[d] for d in scores})==key
        bad={domain:list(keys) for domain,keys in scores.items()}
        bad['X2']=bad['X2'][:-1]
        try:complete_stage(store,result,source,domain_splits,bad,{d:event_ids[d] for d in scores})
        except ValueError:pass
        else:raise AssertionError('Incomplete member predictions accepted')
        partial=store.put_split(dataset,'X2',source.indices('X2')[:-1],
            {'algorithm':'incomplete_fixture','version':1,'seed':7})
        try:complete_stage(store,result,source,{**domain_splits,'X2':partial},scores,{d:event_ids[d] for d in scores})
        except ValueError:pass
        else:raise AssertionError('Incomplete X2 event coverage accepted')
        path=store.records/(result['history_ids'][0]+'.json')
        original=path.read_bytes()
        try:
            changed=json.loads(original);changed['payload'][0]['val_loss']+=1
            path.write_text(json.dumps(changed))
            try:verify_stage_completion(store,key,source)
            except ValueError:pass
            else:raise AssertionError('Corrupt training history accepted')
        finally:path.write_bytes(original)
        assert verify_stage_completion(store,key,source)
        completions.append(key)
        if name is None:
            case_id=f'fixture-step{stage}';case_receipts[case_id]=key
            result_id=registry.publish(case_id,case_tasks[case_id],key)
            assert registry.publish(case_id,case_tasks[case_id],key)==result_id
            assert registry.get(case_id)['completion_id']==key
            if stage==1:
                assert registry.parents('fixture-step2')=={case_id:key}
                wrong=copy.deepcopy(case_tasks[case_id]);wrong['members'][0]['data_seed']+=1
                try:registry.publish(case_id,wrong,key)
                except ValueError:pass
                else:raise AssertionError('Wrong training recipe registered')
                bad_plan=copy.deepcopy(registry_plan);bad_plan['nodes'][0]['max_epochs']=99
                try:CaseRegistry(store,bad_plan,args.out/'case-registry',resume=True)
                except ValueError:pass
                else:raise AssertionError('Registry adopted another plan')
    base=train(store,source,bound(1),args.out/'step1')
    base_scores=export(base,('X1','X2'))
    finish(1,base_scores)
    # Parent completion ordering must not become an implicit encoder pairing.
    base_training=json.loads((args.out/'step1'/'worker'/'training-completion.json').read_text())
    reversed_training={**base_training,'model_ids':base_training['model_ids'][::-1],
                       'history_ids':base_training['history_ids'][::-1]}
    reversed_parent=complete_stage(store,reversed_training,source,{d:splits[d] for d in base_scores},
        {d:keys[::-1] for d,keys in base_scores.items()},{d:event_ids[d] for d in base_scores})
    reordered=materialize_task(store,logical_node(2),pointer,[hparams(2,m) for m in range(2)],
        {'fixture-step1':reversed_parent},expected_dataset=params)
    assert reordered['upstream']['encoder_ids']==base
    incomplete=logical_node(1);incomplete['member_seeds']=[0]
    try:materialize_task(store,incomplete,pointer,[hparams(1,0)],{},expected_dataset=params)
    except ValueError:pass
    else:raise AssertionError('Declared member count was silently reduced')
    wrong_depth=logical_node(1);wrong_depth['depth']={'encoder':99,'decoder':1}
    try:materialize_task(store,wrong_depth,pointer,[hparams(1,m) for m in range(2)],{},expected_dataset=params)
    except ValueError:pass
    else:raise AssertionError('Declared architecture depth was ignored')
    smooth=train(store,source,bound(2,{'encoder_ids':base}),args.out/'step2')
    smooth_scores=export(smooth,('X1','X2'),encoders=base)
    finish(2,smooth_scores)
    region=define_regions(store,base_scores['X1'],event_ids['X1'],smooth_scores['X1'],sr_fraction=.2,cr_fraction=.8,quantile_recipe='sr_quantile_cr_complement_v2')
    bad_node=logical_node(3);bad_node['axes']['eta']='3.0'
    try:materialize_task(store,bad_node,pointer,[hparams(3,m) for m in range(2)],case_receipts,expected_dataset=params)
    except ValueError:pass
    else:raise AssertionError('Wrong-eta parent accepted')
    missing_node=logical_node(2,seeds=[2])
    try:materialize_task(store,missing_node,pointer,[hparams(2,2)],{'fixture-step1':case_receipts['fixture-step1']},expected_dataset=params)
    except ValueError:pass
    else:raise AssertionError('Missing encoder seed accepted')
    unversioned=logical_node(3);unversioned.pop('region_recipe')
    try:materialize_task(store,unversioned,pointer,[hparams(3,m) for m in range(2)],case_receipts,expected_dataset=params)
    except ValueError as exc:assert 'region_recipe' in str(exc)
    else:raise AssertionError('Unversioned Stage-3 task was accepted')
    cr=train(store,source,bound(3,{'region_id':region,'base_X2_scores':base_scores['X2'],'smeared_X2_scores':smooth_scores['X2']}),args.out/'step3')
    cr_scores=export(cr,('X2',))
    finish(3,cr_scores)
    # Existing draft diagnostic: one upstream member, unsmeared representations.
    repr_region=define_regions(store,[base_scores['X1'][1]],event_ids['X1'],[smooth_scores['X1'][1]],sr_fraction=.05,cr_fraction=.95,quantile_recipe='sr_quantile_cr_complement_v2')
    repr_hp={**hparams(2,1),'step':3,'dim_q':6}
    repr_hp.pop('smearing')
    repr_upstream={'region_id':repr_region,'base_X2_scores':[base_scores['X2'][1]],
                   'smeared_X2_scores':[smooth_scores['X2'][1]],'representation_encoder_id':base[1]}
    repr_node=logical_node(3,seeds=[1],sr=.05,model='AttentionClassifier')
    repr_task=materialize_task(store,repr_node,pointer,[repr_hp],
        {key:case_receipts[key] for key in repr_node['requires']},expected_dataset=params)
    assert repr_task['upstream']==repr_upstream
    repr_contexts,_=build_stage_contexts(store,repr_task)
    raw_pair=repr_contexts[0].fetch_train_val_tensor_datasets(FEATURES,'fourTag','weight',training_alignment=32,retain_validation=True)
    repr_pair=repr_contexts[0].fetch_train_val_representation_datasets(FEATURES,'fourTag','weight',device=TEST_DEVICE)
    reference_encoder=load_model(store,base[1],device=TEST_DEVICE)
    for raw_data,encoded_data in zip(raw_pair,repr_pair):
        torch.testing.assert_close(encoded_data.tensors[0],reference_encoder.q_repr(raw_data.tensors[0]),rtol=0,atol=0)
        for raw_tensor,encoded_tensor in zip(raw_data.tensors[1:],encoded_data.tensors[1:]):
            torch.testing.assert_close(raw_tensor,encoded_tensor,rtol=0,atol=0)
    try:build_stage_contexts(store,make_stage_task(pointer,[repr_hp],upstream={**repr_upstream,'region_id':region}))
    except ValueError:pass
    else:raise AssertionError('Main ensemble region accepted for single-upstream diagnostic')
    repr_models=train(store,source,repr_contexts,args.out/'step3-representation')
    repr_scores=export(repr_models,('X2',),encoders=[base[1]])
    finish(3,repr_scores,name='step3-representation')
    ensemble=store.put_ensemble(cr,'mean_probability')
    before=store.storage_stats();log_ratio=store.aggregate_scores(ensemble,cr_scores['X2']);assert store.storage_stats()==before
    groups=classify_X2(store,region,base_scores['X2'],smooth_scores['X2'])
    rows=source.indices('X2');labels=frame.fourTag.values[rows];weights=frame.weight.values[rows]
    inputs,audit=prepare_affine_inputs(np.column_stack([np.zeros(len(rows),dtype=np.int64),rows]),labels,weights,
        groups['log_psi'],log_ratio,dataset_id=dataset,domain='X2',lower=store.read(region)['payload']['log_tau_s'])
    assert audit['positive_normalizers'] and all(np.isfinite(v).all() for v in inputs)
    assert len(cr_scores['X2'])==2 and all(len(store.load_array(key))==len(rows) for key in cr_scores['X2'])
    result={'status':'PASS','scope':'synthetic integration, two members and two epochs per stage; no bootstrap inference','device':TEST_DEVICE,
            'stage_completion_ids':completions,'permanent_histories_verified':True,
            'resumed_worker_matches_reference':True,'completed_training_reused':True,
            'models':{'step1':base,'step2':smooth,'step3':cr,'step3_representation':repr_models},'region':region,'all_X2_rows':len(rows),
            'test_input_counts':{'3b':audit['n3'],'4b':audit['n4']},'storage':store.storage_stats()}
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','stages':3,'members_per_stage':2,'all_X2_rows':len(rows)}),flush=True)


if __name__=='__main__':main()
