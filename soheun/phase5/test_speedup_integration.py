"""Exercise PR #8 against actual artifact training, export and resume APIs."""
import argparse,json,pathlib,sys
from unittest.mock import patch
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'phase5')]
from artifacts.training_store import TrainingStore
from artifacts.campaign_runtime import run_campaign,import_sources
from artifacts.stage_processes import _task
from test_campaign_runtime import fixture
from speedups.scope import execution_patches
from speedups.compare import differences

PATCHES=['nosync','fast_gbn','fast_reinforce','graphs']

def optimized(payload):
    values=list(payload)
    values[4]={**values[4],'execution_patches':PATCHES}
    return _task(tuple(values))


def interrupted(payload):
    values=list(payload)
    values[4]={**values[4],'stop_after_completed_epochs':1}
    return optimized(tuple(values))

def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b);return
    if isinstance(a,np.ndarray):assert np.array_equal(a,b);return
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal(a[key],b[key])
        return
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
        return
    assert a==b

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    plan=fixture(args.out);store=TrainingStore(args.out/'store');import_sources(store,plan)
    options={'device':'cuda','export_batch_size':128,'safety_bytes':1024**3,
             'compute_headroom_bytes':2*1024**3}
    baseline=run_campaign(store,plan,args.out/'baseline',nproc=1,**options)
    candidate=run_campaign(store,plan,args.out/'optimized',nproc=2,execution_patches=PATCHES,**options)
    assert baseline['completion_ids']==candidate['completion_ids']
    for case in baseline['case_ids']:
        a=torch.load(args.out/'baseline/tasks'/case/'training/last.ckpt',map_location='cpu',weights_only=False)
        b=torch.load(args.out/'optimized/tasks'/case/'training/last.ckpt',map_location='cpu',weights_only=False)
        for key in ('state_dict','optimizer_states','lr_schedulers','artifact_training_history'):
            equal(a[key],b[key])
    first=next(n['id'] for n in plan['nodes'] if n['stage']==1)
    with patch('artifacts.campaign_runtime._task',interrupted):
        try:run_campaign(store,plan,args.out/'interrupted',case_ids=[first],nproc=1,execution_patches=PATCHES,**options)
        except RuntimeError:pass
        else:raise AssertionError('Expected a recoverable prefix')
    assert not (args.out/'interrupted/completion.json').exists()
    resumed=run_campaign(store,plan,args.out/'interrupted',case_ids=[first],nproc=1,resume=True,execution_patches=PATCHES,**options)
    try:run_campaign(store,plan,args.out/'interrupted',case_ids=[first],nproc=1,resume=True,**options)
    except ValueError:pass
    else:raise AssertionError('Resume silently changed execution policy')
    assert resumed['completion_ids'][first]==baseline['completion_ids'][first]
    ref={'members':[{'name':'one','state':'a'},{'name':'two','state':'b'}]}
    assert differences(ref,{'members':ref['members'][:1]})==['member_set']
    try:differences(ref,{'members':[ref['members'][0]]*2})
    except ValueError:pass
    else:raise AssertionError('Duplicate fingerprint members accepted')
    import independent_training as it
    original=it.record;cwd=pathlib.Path.cwd()
    with execution_patches(['nosync','fast_reinforce']):assert it.record is not original
    assert it.record is original and pathlib.Path.cwd()==cwd
    manifests=[json.loads(p.read_text()) for p in (args.out/'optimized/tasks').glob('*/training-plan.json')]
    assert all(m['execution_policy']['patches']==PATCHES for m in manifests)
    metrics=[json.loads(p.read_text()) for p in (args.out/'optimized/tasks').glob('*/execution-metrics.json')]
    assert any(m.get('cuda_graphs',{}).get('replays',0)>0 for m in metrics)
    attention_cases=[n['id'] for n in plan['nodes'] if n['stage']==2 or n.get('input_space')=='base_encoder']
    for key in attention_cases:
        metric=json.loads((args.out/'optimized/tasks'/key/'execution-metrics.json').read_text())
        assert metric['cuda_graphs']['captures']>0 and metric['cuda_graphs']['replays']>0
    import stacked_attention_classifier as sa
    original_attention=sa.independent_step
    with execution_patches(PATCHES):assert sa.independent_step is not original_attention
    assert sa.independent_step is original_attention
    result={'status':'PASS','cases':len(baseline['case_ids']),'receipts_exact':True,
            'model_optimizer_scheduler_history_exact':True,'epoch_resume_exact':True,
            'attention_graph_replays_and_scope_restored':True,'attention_history_flush':True,'scope_restored':True,'execution_policy_recorded':True}
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
