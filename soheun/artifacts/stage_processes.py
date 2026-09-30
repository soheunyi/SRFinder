"""Explicit spawn-process coordinator for already-bound independent stage tasks.

Tasks are JSON recipes; the trusted caller supplies a top-level context factory.
No manuscript grid is inferred and no Slurm job is submitted by this module.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import gc
import hashlib
import inspect
import multiprocessing as mp
import time
from pathlib import Path
import torch
from .training_store import TrainingStore,canonical,sha
from .train_stage import _owned_run,_atomic_json
from .run_stage import run_stage
from .memory_policy import worker_budgets


def _initialize():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.autograd.set_detect_anomaly(False)


def _task(payload):
    store_root,spec,factory,output,options=payload
    started=time.perf_counter()
    store=TrainingStore(store_root)
    contexts,source=factory(store,spec)
    prepared=time.perf_counter()
    try:
        result=run_stage(store,contexts,source,output,resume=Path(output).exists(),**options)
        if result['status']!='STAGE_ARTIFACTS_COMPLETE':raise RuntimeError('Worker did not finish its declared stage')
        _atomic_json(Path(output)/'worker-metrics.json',{'context_preparation_s':prepared-started,
            'worker_total_s':time.perf_counter()-started})
        return result
    finally:
        del contexts,source
        gc.collect()
        if options['device']=='cuda':torch.cuda.empty_cache()


def run_tasks(store,tasks,context_factory,output,*,nproc=5,device='cpu',resident='auto',
              safety_bytes=None,compute_headroom_bytes=None,export_batch_size=1024,resume=False,execution_patches=()):
    """Run bound tasks and verify completed tasks again on resume.

    Worker count and placement budgets are execution choices. Task/model recipes
    must remain unchanged. Unexpected failure stops other active workers; their
    last completed-epoch checkpoints remain available for an explicit resume.
    """
    if type(nproc) is not int or nproc<1 or not tasks:raise ValueError('Positive worker count and nonempty tasks required')
    if device not in ('cpu','cuda'):raise ValueError('Unsupported device')
    if not callable(context_factory) or '<locals>' in context_factory.__qualname__:
        raise ValueError('Spawn workers require a top-level context factory')
    source_file=inspect.getsourcefile(context_factory)
    if source_file is None:raise ValueError('Factory needs an inspectable source file')
    factory={'module':context_factory.__module__,'name':context_factory.__qualname__,
             'source_sha256':hashlib.sha256(Path(source_file).read_bytes()).hexdigest()}
    # Canonical serialization rejects tensor/index objects in worker recipes.
    ids=[sha(canonical({'factory':factory,'recipe':spec})) for spec in tasks]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate stage tasks')
    manifest={'schema':1,'tasks':[{'id':key,'recipe':spec} for key,spec in zip(ids,tasks)],
              'factory':factory,'device':device,'export_batch_size':export_batch_size,
              'coordinator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if execution_patches:
        from speedups.policy import descriptor
        manifest['execution_policy']=descriptor(execution_patches)
    root=Path(output)
    with _owned_run(root,manifest,resume):
        count=min(nproc,len(tasks));budgets=[None]*count
        if device=='cuda':
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
                raise ValueError('Allocate exactly one visible GPU for this coordinator')
            if safety_bytes is None or compute_headroom_bytes is None:
                raise ValueError('CUDA coordination needs explicit safety margin and worker headroom')
            budgets=worker_budgets(int(torch.cuda.mem_get_info()[0]),workers=count,safety_bytes=safety_bytes)
            if compute_headroom_bytes>budgets[0]:raise MemoryError('Worker headroom exceeds its allocation budget')
        options={'device':device,'resident':resident,'device_budget_bytes':budgets[0],
                 'compute_headroom_bytes':compute_headroom_bytes,'export_batch_size':export_batch_size,
                 'execution_patches':execution_patches}
        _atomic_json(root/'latest-execution.json',{'requested_workers':nproc,'active_worker_limit':count,
                     'worker_budget_bytes':budgets[0],'options':options})
        pool=ProcessPoolExecutor(max_workers=count,mp_context=mp.get_context('spawn'),initializer=_initialize)
        results={}
        try:
            futures={pool.submit(_task,(str(store.root.resolve()),spec,context_factory,
                      str((root/key).resolve()),options)):key for key,spec in zip(ids,tasks)}
            for future in as_completed(futures):
                key=futures[future];results[key]=future.result()
                _atomic_json(root/'progress.json',{'status':'RUNNING','completed_task_ids':sorted(results),
                                                   'task_count':len(tasks)})
        except BaseException:
            for process in tuple((pool._processes or {}).values()):
                if process.is_alive():process.terminate()
            raise
        finally:pool.shutdown(wait=True,cancel_futures=True)
        result={'status':'BOUND_STAGE_TASKS_COMPLETE','task_ids':ids,
                'stage_completion_ids':[results[key]['completion_id'] for key in ids],
                'results':[results[key] for key in ids]}
        _atomic_json(root/'completion.json',result)
        return result
