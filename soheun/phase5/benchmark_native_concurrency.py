"""Bounded native comparison: five CR ensembles, serial vs five workers plus restart.

Uses existing verified upstreams and the declared draft plan. Never launches a
campaign. Both arms use identical patches, MPS, export profiles and model recipes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import hashlib,json,multiprocessing as mp,os,threading,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'phase5')]
import torch
from artifacts.training_store import TrainingStore,canonical,sha
from artifacts.case_registry import CaseRegistry
from artifacts.bound_tasks import build_stage_contexts
from artifacts.materialize_task import materialize_task
from artifacts.campaign_recipes import recipes
from artifacts.memory_policy import worker_budgets
from artifacts.stage_processes import _initialize
from artifacts.run_stage import run_stage
from artifacts.train_stage import _atomic_json
from test_speedup_integration import equal
PATCHES=['nosync','fast_gbn','fast_reinforce','graphs']


def overlay(source,destination):
    target=TrainingStore(destination)
    for src,dst in ((source.records,target.records),(source.blobs,target.blobs)):
        for path in src.iterdir():
            if path.is_file() and not path.name.startswith('.pending-'):os.link(path,dst/path.name)
    return target


@contextmanager
def timing():
    """Epoch-level validation sync and checkpoint I/O timing; no minibatch sync."""
    import pytorch_lightning as pl
    import phase3.resumable as recovery
    totals={'validation_s':0.,'checkpoint_s':0.,'checkpoint_calls':0,
            'instrumentation':'CUDA sync at validation boundaries; host wall time for checkpoint writes'}
    class ValidationTimer(pl.Callback):
        def on_validation_start(self,trainer,module):
            torch.cuda.synchronize();self.started=time.perf_counter()
        def on_validation_end(self,trainer,module):
            torch.cuda.synchronize();totals['validation_s']+=time.perf_counter()-self.started
    trainer=pl.Trainer;save=recovery.atomic_torch_save
    def measured(function):
        def invoke(*args,**kwargs):
            started=time.perf_counter()
            try:return function(*args,**kwargs)
            finally:
                totals['checkpoint_s']+=time.perf_counter()-started;totals['checkpoint_calls']+=1
        return invoke
    class TimedTrainer(trainer):
        def __init__(self,**kwargs):
            kwargs['callbacks']=[ValidationTimer(),*kwargs.get('callbacks',[])]
            super().__init__(**kwargs)
    pl.Trainer=TimedTrainer
    recovery.atomic_torch_save=measured(save)
    try:yield totals
    finally:
        pl.Trainer=trainer;recovery.atomic_torch_save=save


def worker(payload):
    store_path,task,path,phase,budget=payload
    started=time.perf_counter();store=TrainingStore(store_path)
    contexts,source=build_stage_contexts(store,task);prepared=time.perf_counter()
    output=Path(path)
    with timing() as timers:
        result=run_stage(store,contexts,source,output,device='cuda',resident='auto',
            resume=output.exists(),export_batch_size=32768,
            stop_after_completed_epochs=16 if phase=='prefix' else None,
            device_budget_bytes=budget,compute_headroom_bytes=5*1024**3,execution_patches=PATCHES)
    expected='INTERRUPTED_RECOVERABLE' if phase=='prefix' else 'STAGE_ARTIFACTS_COMPLETE'
    if result['status']!=expected:raise ValueError('Unexpected training completion boundary')
    if phase=='prefix' and result['completed_epochs']!=16:raise ValueError('Incorrect restart boundary')
    totals={'phase':phase,'context_preparation_s':prepared-started,
            'wall_s':time.perf_counter()-started,**timers,
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved()}
    import resource
    totals['peak_host_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record=result if phase=='prefix' else json.loads((output/'training/training-completion.json').read_text())
    for key in ('preparation_s','fit_checkpoint_s'):totals[key]=record[key]
    # Validation callback runs before model-file checkpoint callbacks. Trainer
    # checkpoint writes are measured separately; remaining fit time includes dispatch.
    totals['training_and_dispatch_s']=record['fit_checkpoint_s']-timers['validation_s']-timers['checkpoint_s']
    if phase!='prefix':totals['stage_metrics']=json.loads((output/'execution-metrics.json').read_text())
    _atomic_json(output/(phase+'-timing.json'),totals)
    return {'case_id':task['logical_case_id'],'result':result,'timing':totals}


def monitor(stop,path):
    import psutil,subprocess
    parent=psutil.Process();gpu_uuid=getattr(torch.cuda.get_device_properties(0),'uuid',None)
    with path.open('w') as log:
        while not stop.is_set():
            row={'time':time.time(),'allocated_gpu_uuid':str(gpu_uuid) if gpu_uuid else None,
                 'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
            try:
                processes=[parent,*parent.children(recursive=True)]
                row['aggregate_host_rss_bytes']=sum(p.memory_info().rss for p in processes if p.is_running())
                # Preserve UUIDs for mapping; never interpret a global GPU index as CUDA ordinal.
                r=subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                row['visible_gpu_csv']=r.stdout.strip();row['gpu_query_returncode']=r.returncode
            except (OSError,psutil.Error,subprocess.TimeoutExpired) as error:row['error']=str(error)
            log.write(json.dumps(row)+'\n');log.flush();stop.wait(5)


def prepare(args):
    args.out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(args.plan.read_text())
    wanted={(1.,.05),(1.,.10),(1.,.15),(1.,.20),(float('inf'),.20)}
    nodes=[n for n in plan['nodes'] if n['stage']==3 and n['member_count']==5
           and n['axes']['signal']=='HH4b' and float(n['axes']['epsilon'])==.02
           and n['axes']['mother_seed']==1
           and (float(n['axes']['eta']),float(n['axes']['sr_fraction'])) in wanted]
    if len(nodes)!=5 or any(n['max_epochs']!=100 for n in nodes):raise ValueError('Bounded five-ensemble scope changed')
    original=TrainingStore(args.store)
    oldplan=json.loads((args.execution/'frozen-plan.json').read_text())
    registry=CaseRegistry(original,oldplan,args.execution/'registry',resume=True)
    parents={key:registry.get(key) for n in nodes for key in n['requires']}
    if any(v is None for v in parents.values()):raise ValueError('Required native upstream missing')
    tasks=[]
    seed=overlay(original,args.out/'seed-store')
    for node in nodes:
        pointer=plan['sources'][node['source_case_id']]['source']
        tasks.append(materialize_task(seed,node,pointer,recipes(plan,node),
            {key:parents[key]['completion_id'] for key in node['requires']},expected_dataset=pointer['hparams']['dataset']))
    for arm in ('serial','parallel'):overlay(seed,args.out/arm/'store')
    _atomic_json(args.out/'benchmark-plan.json',{'schema':1,'plan_sha256':sha(canonical(plan)),
        'tasks':tasks,'patches':PATCHES,'export_batch_size':32768,'members':25,'epochs':100,
        'restart_after_epoch_count':16,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations':['Five distinct CR selections from one mother sample; not five distinct mother samples',
                       'Both arms use MPS and the same patches; this isolates worker concurrency',
                       'Parallel wall time includes deliberate restart overhead; one timed trial per arm']})
    print('PREPARED: five native CR ensembles, 25 members; no training yet',flush=True)


def execute(args,manifest):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Allocate one visible GPU')
    if not os.environ.get('CUDA_MPS_PIPE_DIRECTORY'):raise ValueError('Use the private MPS wrapper')
    phase=args.phase;arm='serial' if phase=='serial' else 'parallel';count=1 if phase=='serial' else 5
    budget=worker_budgets(int(torch.cuda.mem_get_info()[0]),workers=count,safety_bytes=3*1024**3)[0]
    if budget<5*1024**3:raise MemoryError('Five-GiB per-worker compute headroom does not fit')
    root=args.out/arm;stop=threading.Event()
    watcher=threading.Thread(target=monitor,args=(stop,root/(phase+'-resources.jsonl')),daemon=True)
    watcher.start();started=time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=count,mp_context=mp.get_context('spawn'),initializer=_initialize) as pool:
            futures=[pool.submit(worker,(str(root/'store'),task,str(root/'tasks'/task['logical_case_id']),phase,budget)) for task in manifest['tasks']]
            results=[f.result() for f in futures]
        _atomic_json(root/(phase+'-report.json'),{'phase':phase,'wall_s':time.perf_counter()-started,'workers':count,'results':results})
    finally:stop.set();watcher.join(timeout=10)


def compare(args,manifest):
    serial=json.loads((args.out/'serial/serial-report.json').read_text())
    prefix=json.loads((args.out/'parallel/prefix-report.json').read_text())
    resumed=json.loads((args.out/'parallel/resume-report.json').read_text())
    by_case={r['case_id']:r['result'] for r in serial['results']}
    for row in resumed['results']:
        key=row['case_id']
        if row['result']['completion_id']!=by_case[key]['completion_id']:raise ValueError('Best models/full-domain score artifacts differ')
        a,b=[torch.load(args.out/arm/'tasks'/key/'training/last.ckpt',map_location='cpu',weights_only=False) for arm in ('serial','parallel')]
        for field in ('state_dict','optimizer_states','lr_schedulers','artifact_training_history'):equal(a[field],b[field])
    parallel_s=prefix['wall_s']+resumed['wall_s']
    report={'status':'PASS_NATIVE_FIVE_ENSEMBLE_CONCURRENCY_AND_RESUME','members':25,'epochs':100,
        'serial_wall_s':serial['wall_s'],'parallel_wall_s_including_restart':parallel_s,
        'speedup':serial['wall_s']/parallel_s,
        'members_per_gpu_hour':{'serial':25*3600/serial['wall_s'],'five_workers':25*3600/parallel_s},
        'resource_sampling':'aggregate RSS sums coordinator and worker RSS every five seconds; shared pages may be counted in multiple processes',
        'exact_fields':['best models and full X2 score receipts','final weights and normalization','Adam','schedulers','loss histories'],
        'phases':{'serial':serial,'prefix':prefix,'resume':resumed},'limitations':manifest['limitations']}
    _atomic_json(args.out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='phases'}),flush=True)


def main():
    ap=argparse.ArgumentParser()
    for key in ('plan','execution','store','out'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--phase',choices=['prepare','serial','prefix','resume','compare'],required=True)
    args=ap.parse_args();_initialize()
    if args.phase=='prepare':prepare(args);return
    manifest=json.loads((args.out/'benchmark-plan.json').read_text())
    if manifest['source_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise ValueError('Benchmark changed')
    if args.phase=='compare':compare(args,manifest)
    else:execute(args,manifest)

if __name__=='__main__':main()
