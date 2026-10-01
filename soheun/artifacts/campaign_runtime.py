"""Bounded execution of an explicitly supplied dependency plan.

This module never infers a sweep, submits Slurm jobs, chooses an aggregation
rule, deletes checkpoints or declares scientific acceptance. The caller owns
execution authorization, allocation, prepared inputs and deployment snapshot.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import torch
from .training_store import TrainingStore, canonical, sha
from .train_stage import _owned_run, _atomic_json
from .case_registry import CaseRegistry
from .campaign_recipes import recipes
from .bound_tasks import build_stage_contexts
from .materialize_task import materialize_task
from .stage_processes import _initialize, _task
from .memory_policy import worker_budgets


def runtime_snapshot():
    root = Path(__file__).resolve().parents[1]
    # Freeze dependency/reader code as well as the optimizer/model implementation.
    paths = sorted(root.rglob('*.py'))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if not {'__pycache__','phase5_runs','.git'} & set(p.relative_to(root).parts)}


def import_sources(store, plan):
    origin_path = Path(plan['source_store'])
    if not (origin_path / 'records').is_dir():
        raise FileNotFoundError('Verified source store is unavailable')
    origin = TrainingStore(origin_path)
    declared = {row['source']['dataset_id'] for row in plan['sources'].values()}
    # Only needed immutable descriptors; splits are reconstructed by their recipes.
    for key in sorted(declared):
        record = origin.read(key, 'dataset')
        if record['payload'] is not None:
            raise ValueError('Source descriptors must not contain event payloads')
        store._publish(store.records / (key + '.json'), canonical(record))
        store.read(key, 'dataset')
    return len(declared)


def selected_nodes(plan, case_ids=None):
    nodes = {node['id']: node for node in plan['nodes']}
    if len(nodes) != len(plan['nodes']):
        raise ValueError('Duplicate logical cases')
    for node in nodes.values():
        if any(key not in nodes or nodes[key]['stage'] >= node['stage'] for key in node['requires']):
            raise ValueError('Invalid or cyclic dependency plan')
    wanted = set(nodes) if case_ids is None else set(case_ids)
    if not wanted or wanted - nodes.keys():
        raise ValueError('Requested case is outside the frozen plan')
    pending = list(wanted)
    while pending:
        key = pending.pop()
        for parent in nodes[key]['requires']:
            if parent not in wanted:
                wanted.add(parent)
                pending.append(parent)
    return [node for node in plan['nodes'] if node['id'] in wanted]



def execution_manifest(store, plan, *, case_ids=None, device='cpu', export_batch_size=1024, execution_patches=()):
    settings=plan.get('fixed_settings',{})
    expected_version=settings.get('torch_version')
    if 'sources' in plan and expected_version is None:
        raise ValueError('Production plan must pin the torch version')
    if expected_version is not None and str(torch.__version__)!=expected_version:
        raise ValueError('Torch runtime differs from the frozen production plan')
    if settings.get('adam_epsilon',1e-8)!=1e-8:
        raise ValueError('Campaign Adam epsilon differs from explicit optimizer construction')
    nodes = selected_nodes(plan, case_ids)
    snapshot = runtime_snapshot()
    result = {'schema': 1, 'plan_sha256': sha(canonical(plan)),
        'runtime_sha256': sha(canonical(snapshot)), 'runtime_files': snapshot,
        'store': str(store.root.resolve()), 'case_ids': sorted(n['id'] for n in nodes),
        'device': device, 'export_batch_size': export_batch_size}
    if execution_patches:
        from speedups.policy import descriptor
        result['execution_policy']=descriptor(execution_patches)
    return result


def prepare_execution(store, plan, output, *, case_ids=None, device='cpu',
                      export_batch_size=1024, resume=False, execution_patches=()):
    """Freeze input/ownership and import descriptors; no model, CUDA or job start."""
    manifest = execution_manifest(store, plan, case_ids=case_ids, device=device,
                                  export_batch_size=export_batch_size, execution_patches=execution_patches)
    root = Path(output)
    with _owned_run(root, manifest, resume):
        imported = import_sources(store, plan)
        store._publish(root / 'frozen-plan.json', canonical(plan))
        report = {'status': 'PREPARED_TRAINING_NOT_STARTED',
            'plan_sha256': manifest['plan_sha256'], 'runtime_sha256': manifest['runtime_sha256'],
            'declared_cases': len(manifest['case_ids']), 'source_descriptors': imported,
            'pending_gates': plan.get('gates', []), 'aggregation': plan.get('aggregation')}
        store._publish(root / 'prepared.json', canonical(report))
        return report


def run_campaign(store, plan, output, *, case_ids=None, nproc=5, device='cpu',
                 resident='auto', safety_bytes=None, compute_headroom_bytes=None,
                 export_batch_size=1024, resume=False, max_new_cases=None, execution_patches=(), through_stage=3, work_case_ids=None):
    """Execute only a declared plan/closure with at most nproc tasks in flight.

    A max_new_cases prefix stops cleanly after complete training/export/registry
    transactions. It does not simulate mid-epoch recovery. Worker count and
    placement may change on resume; plan, scope, code and export profile may not.
    through_stage and work_case_ids bound scheduling without changing the full
    declared scope, so pilot results can be reused by the later full campaign.
    """
    if type(through_stage) is not int or through_stage not in (1,2,3):
        raise ValueError('through_stage must be 1, 2 or 3')
    if type(nproc) is not int or nproc < 1:
        raise ValueError('Positive worker count required')
    if max_new_cases is not None and (type(max_new_cases) is not int or max_new_cases < 1):
        raise ValueError('Positive completion limit required')
    if device not in ('cpu', 'cuda'):
        raise ValueError('Unsupported device')
    nodes = selected_nodes(plan, case_ids)
    root = Path(output)
    manifest = execution_manifest(store, plan, case_ids=case_ids, device=device,
                                  export_batch_size=export_batch_size, execution_patches=execution_patches)
    snapshot = manifest['runtime_files']
    with _owned_run(root, manifest, resume):
        if (root / 'frozen-plan.json').is_file():
            if json.loads((root / 'frozen-plan.json').read_text()) != plan:
                raise ValueError('Frozen input plan changed')
        else: store._publish(root / 'frozen-plan.json', canonical(plan))
        registry = CaseRegistry(store, plan, root / 'registry', resume=(root / 'registry').exists())
        (root / 'tasks').mkdir(exist_ok=True)
        done = {}
        for node in nodes:
            value = registry.get(node['id'])
            if value is not None:
                done[node['id']] = value['completion_id']
        active_nodes=nodes if work_case_ids is None else selected_nodes(plan,work_case_ids)
        if {n['id'] for n in active_nodes}-{n['id'] for n in nodes}:
            raise ValueError('Work selection lies outside the prepared execution scope')
        active_nodes=[n for n in active_nodes if n['stage']<=through_stage]
        remaining = [node for node in active_nodes if node['id'] not in done]
        limit = len(remaining) if max_new_cases is None else min(max_new_cases, len(remaining))
        workers = min(nproc, max(1, limit))
        budget = None
        if device == 'cuda':
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise ValueError('Allocate exactly one visible GPU')
            if safety_bytes is None or compute_headroom_bytes is None:
                raise ValueError('CUDA needs explicit safety and per-worker compute headroom')
            budget = worker_budgets(int(torch.cuda.mem_get_info()[0]), workers=workers,
                                    safety_bytes=safety_bytes)[0]
            if compute_headroom_bytes > budget:
                raise MemoryError('Worker headroom exceeds its allocation budget')
        options = {'device': device, 'resident': resident, 'device_budget_bytes': budget,
                   'compute_headroom_bytes': compute_headroom_bytes,
                   'export_batch_size': export_batch_size, 'execution_patches': execution_patches}
        _atomic_json(root / 'latest-execution.json', {'requested_workers': nproc,
            'active_worker_limit': workers, 'options': options, 'max_new_cases': max_new_cases, 'through_stage': through_stage,
            'work_case_ids':None if work_case_ids is None else sorted(work_case_ids),
            'scheduled_case_count':len(active_nodes)})
        started = time.monotonic()
        new_count = 0
        in_flight = {}
        max_in_flight = 0
        pool = None
        def progress(status, **extra):
            _atomic_json(root / 'progress.json', {'status': status, 'total_cases': len(nodes),
                'completed_cases': len(done), 'newly_completed_cases': new_count,
                'running_case_ids': sorted(node['id'] for node, _ in in_flight.values()),
                'max_in_flight': max_in_flight, 'elapsed_s': time.monotonic() - started, **extra})
        progress('RUNNING')
        try:
            if limit:
                pool = ProcessPoolExecutor(max_workers=workers,
                    mp_context=mp.get_context('spawn'), initializer=_initialize)
            while remaining or in_flight:
                while pool is not None and len(in_flight) < workers and new_count + len(in_flight) < limit:
                    ready = next((node for node in remaining if all(p in done for p in node['requires'])), None)
                    if ready is None:
                        break
                    pointer = plan['sources'][ready['source_case_id']]['source']
                    parents = registry.parents(ready['id'])
                    if parents is None:
                        raise RuntimeError('Dependency result is not registered')
                    task = materialize_task(store, ready, pointer, recipes(plan, ready), parents,
                                            expected_dataset=pointer['hparams']['dataset'])
                    future = pool.submit(_task, (str(store.root.resolve()), task, build_stage_contexts,
                        str((root / 'tasks' / ready['id']).resolve()), options))
                    in_flight[future] = (ready, task)
                    remaining.remove(ready)
                    max_in_flight = max(max_in_flight, len(in_flight))
                progress('RUNNING')
                if not in_flight:
                    if new_count == limit:
                        break
                    raise RuntimeError('Dependency plan cannot make progress')
                completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in completed:
                    node, task = in_flight.pop(future)
                    result = future.result()
                    registry.publish(node['id'], task, result['completion_id'])
                    verified = registry.get(node['id'])
                    done[node['id']] = verified['completion_id']
                    new_count += 1
                    progress('RUNNING', last_completed_case=node['id'])
                # Reject changing a deployed program during execution as on resume.
                if runtime_snapshot() != snapshot:
                    raise ValueError('Runtime code changed during execution')
            status = 'CAMPAIGN_ARTIFACTS_COMPLETE' if len(done) == len(nodes) else 'CAMPAIGN_PREFIX_COMPLETE'
            result = {'status': status, 'plan_sha256': manifest['plan_sha256'],
                'runtime_sha256': manifest['runtime_sha256'], 'case_ids': manifest['case_ids'],
                'completion_ids': done, 'max_in_flight': max_in_flight,
                'selected_scope_complete':all(n['id'] in done for n in active_nodes),
                'scheduled_case_count':len(active_nodes)}
            _atomic_json(root / 'completion.json', result)
            progress(status)
            return result
        except BaseException as exc:
            progress('FAILED', error_type=type(exc).__name__, error=str(exc))
            if pool is not None:
                for process in tuple((pool._processes or {}).values()):
                    if process.is_alive():
                        process.terminate()
            raise
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
