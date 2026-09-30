"""Measure native 15-member Steps 1/2 with integrated graphs and full exports."""
import argparse,json,os,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'phase5')]
from artifacts.training_store import TrainingStore
from artifacts.case_registry import CaseRegistry
from artifacts.bound_tasks import build_stage_contexts
from artifacts.materialize_task import materialize_task
from artifacts.campaign_recipes import recipes
from artifacts.run_stage import run_stage
from test_speedup_integration import equal
from benchmark_native_concurrency import timing,PATCHES


def main():
    ap=argparse.ArgumentParser()
    for key in ('execution','store','out'):ap.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not os.environ.get('CUDA_MPS_PIPE_DIRECTORY'):raise ValueError('Use the private MPS wrapper')
    args.out.mkdir(parents=True,exist_ok=False)
    original=TrainingStore(args.store);store=TrainingStore(args.out/'store')
    for src,dst in ((original.records,store.records),(original.blobs,store.blobs)):
        for p in src.iterdir():
            if p.is_file() and not p.name.startswith('.pending-'):os.link(p,dst/p.name)
    plan=json.loads((args.execution/'frozen-plan.json').read_text())
    registry=CaseRegistry(original,plan,args.execution/'registry',resume=True)
    declared=json.loads((args.execution/'training-plan.json').read_text())['case_ids']
    report={'status':'RUNNING','stages':[],'limitations':['One native ensemble per stage, not five concurrent 15-member ensembles','Fits include timing instrumentation; exports use batch 32768']}
    produced={}
    for stage in (1,2):
        keys=[key for key in declared if registry.nodes[key]['stage']==stage]
        if len(keys)!=1:raise ValueError('Expected one reference ensemble per upstream stage')
        key=keys[0];reference=registry.get(key);node=registry.nodes[key]
        if node['member_count']!=15 or node['max_epochs']!={1:100,2:30}[stage]:raise ValueError('Native scope changed')
        pointer=reference['task']['source']
        task=materialize_task(store,node,pointer,recipes(plan,node),
            {parent:produced[parent] for parent in node['requires']},expected_dataset=pointer['hparams']['dataset'])
        contexts,source=build_stage_contexts(store,task);started=time.perf_counter()
        with timing() as times:
            result=run_stage(store,contexts,source,args.out/f'step{stage}',device='cuda',resident='auto',
                device_budget_bytes=40*1024**3,compute_headroom_bytes=5*1024**3,
                export_batch_size=32768,execution_patches=PATCHES)
        wall=time.perf_counter()-started
        if result['status']!='STAGE_ARTIFACTS_COMPLETE':raise ValueError('Incomplete upstream stage')
        produced[key]=result['completion_id']
        a=torch.load(args.execution/'tasks'/key/'training/last.ckpt',map_location='cpu',weights_only=False)
        b=torch.load(args.out/f'step{stage}'/'training/last.ckpt',map_location='cpu',weights_only=False)
        for field in ('state_dict','optimizer_states','lr_schedulers','artifact_training_history'):equal(a[field],b[field])
        old=reference['completion']
        for previous,current in zip(old['model_ids'],result['model_ids']):
            if original.read(previous)['payload']!=store.read(current)['payload']:raise ValueError('Best weights differ')
        for domain in ('X1','X2'):
            for previous,current in zip(old['score_ids'][domain],result['score_ids'][domain]):
                if original.read(previous)['payload']!=store.read(current)['payload']:raise ValueError('Prediction arrays differ')
        metrics=json.loads((args.out/f'step{stage}'/'execution-metrics.json').read_text())
        row={'stage':stage,'members':15,'epochs':node['max_epochs'],'wall_s':wall,
             'members_per_gpu_hour_including_export':15*3600/wall,'exact_reference_states_best_weights_and_scores':True,
             'metrics':metrics,'timing_breakdown':times}
        report['stages'].append(row)
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
    report['status']='PASS_NATIVE_UPSTREAMS_WITH_ATTENTION_GRAPHS'
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
