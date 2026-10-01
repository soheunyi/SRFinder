"""Five -> fifteen trains only new seeds and preserves all original artifacts."""
import argparse,fcntl,json,sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'phase5')]
from test_campaign_runtime import fixture,interrupted_task
from artifacts.campaign_runtime import run_campaign,import_sources
from artifacts.training_store import TrainingStore,canonical,sha
from artifacts.case_registry import CaseRegistry
from artifacts.stage_processes import _initialize
from artifacts.member_extension import make_extension_plan,origin_guard,append_stage_members,combine_completions
from artifacts.stage_completion import complete_stage
from artifacts.bound_tasks import build_stage_contexts
from artifacts.cleanup_completed import prune_completed_case
from artifacts.evaluation import open_reader


def rejects(fn):
    try:fn()
    except (ValueError,RuntimeError):return
    raise AssertionError('Invalid extension accepted')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);_initialize()
    plan=fixture(args.out)
    raw=next(n for n in plan['nodes'] if n['stage']==3 and n['member_count']==2)
    # One real synthetic five-member CR ensemble, paired with two tiny upstreams.
    for node in plan['nodes']:
        if node['stage']==3 and node.get('purpose')!='original_vs_representation':
            node['member_count']=5;node['member_seeds']=list(range(5))
    store=TrainingStore(args.out/'store');import_sources(store,plan)
    options={'device':args.device,'resident':'auto','export_batch_size':128,
             'safety_bytes':1024**3,'compute_headroom_bytes':2*1024**3}
    if args.device=='cuda':options['execution_patches']=('nosync','fast_gbn','fast_reinforce','graphs')
    origin=args.out/'original'
    run_campaign(store,plan,origin,case_ids=[raw['id']],nproc=1,**options)
    original_registry=CaseRegistry(store,plan,origin/'registry',resume=True)
    previous=original_registry.get(raw['id'])
    records=[*previous['completion']['model_ids'],*previous['completion']['history_ids'],*previous['completion']['score_ids']['X2']]
    snapshots={key:canonical(store.read(key)) for key in records}
    original_files={str(p.relative_to(origin)):p.read_bytes() for p in origin.rglob('*') if p.is_file() and p.suffix!='.lock'}
    manifest=json.loads((origin/'training-plan.json').read_text())
    decision={'status':'USER_DECISION_RECORDED','step3_member_count':15,'decision_reference':'synthetic fixture only'}
    extended=make_extension_plan(plan,manifest,origin,member_count=15,decision=decision)
    assert plan['nodes'][plan['nodes'].index(raw)]['member_count']==5
    assert extended['nodes'][plan['nodes'].index(raw)]['member_count']==15
    rejects(lambda:make_extension_plan(plan,manifest,origin,member_count=15,decision={}))
    wrong=deepcopy(extended);wrong['templates']['raw_cr']['optimizer']['lr']=.02
    def check_guard(value):
        with origin_guard(store,value):pass
    rejects(lambda:check_guard(wrong))
    with (origin/'.worker.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);rejects(lambda:check_guard(extended))
    output=args.out/'extended'
    done=run_campaign(store,extended,output,case_ids=[raw['id']],nproc=1,**options)
    target=CaseRegistry(store,extended,output/'registry',resume=True).get(raw['id'])
    assert done['status']=='CAMPAIGN_ARTIFACTS_COMPLETE'
    assert target['completion']['model_ids'][:5]==previous['completion']['model_ids']
    assert target['completion']['history_ids'][:5]==previous['completion']['history_ids']
    assert target['completion']['score_ids']['X2'][:5]==previous['completion']['score_ids']['X2']
    marker=json.loads((output/'tasks'/raw['id']/'stage-completion.json').read_text())
    proof=store.read(marker['extension_record_id'],'member_extension')['identity']
    component=store.read(proof['added_completion_id'],'stage_completion')['identity']
    assert [store.read(m,'model')['identity']['training_recipe']['hparams']['model_seed'] for m in component['model_ids']]==list(range(5,15))
    assert len(component['model_ids'])==10 and len(target['completion']['model_ids'])==15
    contexts,source=build_stage_contexts(store,target['task'])
    old=previous['completion']
    reversed_training={**old,'status':'TRAINING_COMPLETE_EXPORT_PENDING','model_ids':old['model_ids'][::-1],'history_ids':old['history_ids'][::-1]}
    reversed_id=complete_stage(store,reversed_training,source,old['evaluation_splits'],{'X2':old['score_ids']['X2'][::-1]},old['event_metadata_ids'])
    assert combine_completions(store,contexts,source,reversed_id,proof['added_completion_id'])==target['completion_id']
    for key,value in snapshots.items():assert canonical(store.read(key))==value
    for name,value in original_files.items():assert (origin/name).read_bytes()==value
    # A fresh fifteen-member fit is the independence/equivalence reference.
    reference_plan=deepcopy(extended);reference_plan.pop('member_extension')
    reference=run_campaign(store,reference_plan,args.out/'reference',case_ids=[raw['id']],nproc=1,**options)
    assert reference['completion_ids'][raw['id']]==target['completion_id']
    interrupted=args.out/'interrupted'
    with patch('artifacts.campaign_runtime._task',interrupted_task):
        try:run_campaign(store,extended,interrupted,case_ids=[raw['id']],nproc=1,**options)
        except RuntimeError:pass
        else:raise AssertionError('Expected append interruption')
    checkpoint=torch.load(interrupted/'tasks'/raw['id']/'added/training/last.ckpt',map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==0 and len(checkpoint['artifact_training_history'])==1
    resumed=run_campaign(store,extended,interrupted,case_ids=[raw['id']],nproc=2,resume=True,**options)
    assert resumed['completion_ids'][raw['id']]==target['completion_id']
    for key,value in snapshots.items():assert canonical(store.read(key))==value
    with patch('artifacts.campaign_runtime.ProcessPoolExecutor',side_effect=AssertionError('Unexpected refit')):
        assert run_campaign(store,extended,output,case_ids=[raw['id']],nproc=2,resume=True,**options)['completion_ids']==done['completion_ids']
    dry=prune_completed_case(store,output,raw['id'])
    assert len(dry['files'])==21 and all('/added/training/' in r['path'] for r in dry['files'])
    cleaned=prune_completed_case(store,output,raw['id'],apply=True)
    assert cleaned['applied'] and len(cleaned['files'])==21
    reader,_=open_reader(output);values,audit=reader.aggregate(raw['id'],'X2',aggregation='mean_probability')
    assert len(audit['model_ids'])==15 and len(values)>0
    contexts,source=build_stage_contexts(store,target['task'])
    with patch('pytorch_lightning.Trainer.fit',side_effect=AssertionError('Unexpected refit after cleanup')):
        reused=append_stage_members(store,contexts,source,output/'tasks'/raw['id'],
            original_completion_id=previous['completion_id'],device=args.device,export_batch_size=128,
            resident='auto',resume=True,execution_patches=options.get('execution_patches',()))
    assert reused['completion_id']==target['completion_id']
    result={'status':'PASS','device':args.device,'only_seeds_5_to_14_trained':True,
        'original_artifacts_unchanged':True,'fifteen_member_reference_exact':True,
        'upstream_reused':True,'append_epoch_resume_exact':True,'completed_resume_without_refit':True,'extension_cleanup_verified':True}
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
