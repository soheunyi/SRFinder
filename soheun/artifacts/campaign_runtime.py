"""Bounded execution of an explicitly supplied dependency plan.

This module never infers a sweep, submits Slurm jobs, chooses an aggregation
rule, deletes checkpoints or declares scientific acceptance. The caller owns
execution authorization, allocation, prepared inputs and deployment snapshot.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import fcntl
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
from .member_extension import origin_guard,import_unchanged_cases


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



def shard_sources(nodes, shard):
    """Deterministic, identity-free assignment of whole sources to one shard.

    Every training chain is self-contained per source (Step 2 needs only Step 1
    of its source, Step 3 only Steps 1-2), so source shards never wait on each
    other. Sources are ordered by id; shard k of n takes indices i % n == k.
    """
    k, n = shard
    if type(k) is not int or type(n) is not int or n < 2 or not 0 <= k < n:
        raise ValueError('Shard must be (k, n) with n >= 2 and 0 <= k < n')
    by_id = {node['id']: node for node in nodes}
    for node in nodes:
        for parent in node['requires']:
            if parent in by_id and by_id[parent]['source_case_id'] != node['source_case_id']:
                raise ValueError('Cross-source dependency; source sharding would be unsafe')
    sources = sorted({node['source_case_id'] for node in nodes})
    return [source for i, source in enumerate(sources) if i % n == k]


@contextmanager
def _shared_run(root, manifest, shard, sources):
    """Shared ownership of a prepared execution for one shard coordinator.

    Shard coordinators hold the execution lock in shared mode, so an unsharded
    coordinator (exclusive) and shards exclude each other. Each shard also
    holds an exclusive lock for its (k, n) and for every source it owns, so a
    duplicate or overlapping shard specification fails before training.
    """
    root = Path(root)
    if not (root / 'prepared.json').is_file():
        raise ValueError('Sharded execution requires a prepared execution root')
    with ExitStack() as stack:
        lock = stack.enter_context((root / '.worker.lock').open('a+'))
        try: fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Execution has an active unsharded coordinator') from exc
        path = root / 'training-plan.json'
        if not path.is_file() or json.loads(path.read_text()) != manifest:
            raise ValueError('Training output has a different recipe or source version')
        claims = root / 'claims'
        claims.mkdir(exist_ok=True)
        k, n = shard
        mine = stack.enter_context((claims / f'shard-{k}-of-{n}.lock').open('a+'))
        try: fcntl.flock(mine, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Shard {k}/{n} already has an active coordinator') from exc
        for source in sources:
            held = stack.enter_context((claims / f'source-{source}.lock').open('a+'))
            try: fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'Source {source} is claimed by another coordinator') from exc
        yield


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
                 export_batch_size=1024, resume=False, max_new_cases=None, execution_patches=(), through_stage=3, work_case_ids=None,
                 shard=None):
    """Execute only a declared plan/closure with at most nproc tasks in flight.

    A max_new_cases prefix stops cleanly after complete training/export/registry
    transactions. It does not simulate mid-epoch recovery. Worker count and
    placement may change on resume; plan, scope, code and export profile may not.
    through_stage and work_case_ids bound scheduling without changing the full
    declared scope, so pilot results can be reused by the later full campaign.
    shard=(k, n) runs one of n concurrent coordinators (one GPU each) on the
    same prepared execution and registry; it schedules only its own sources and
    never changes a case, task or model identity.
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
    if shard is not None:
        if plan.get('member_extension'):
            raise ValueError('Sharding is not supported for member-extension plans')
        assigned = shard_sources(nodes, shard)
        ownership = _shared_run(root, manifest, shard, assigned)
        suffix = f'-shard-{shard[0]}-of-{shard[1]}'
    else:
        assigned = None
        ownership = _owned_run(root, manifest, resume)
        suffix = ''
    with ownership, origin_guard(store,plan) as origin:
        if (root / 'frozen-plan.json').is_file():
            if json.loads((root / 'frozen-plan.json').read_text()) != plan:
                raise ValueError('Frozen input plan changed')
        else: store._publish(root / 'frozen-plan.json', canonical(plan))
        registry = CaseRegistry(store, plan, root / 'registry',
                                resume=shard is not None or (root / 'registry').exists())
        (root / 'tasks').mkdir(exist_ok=True)
        imported=[] if origin is None else import_unchanged_cases(registry,origin,nodes)
        done = {}
        for node in nodes:
            value = registry.get(node['id'])
            if value is not None:
                done[node['id']] = value['completion_id']
        active_nodes=nodes if work_case_ids is None else selected_nodes(plan,work_case_ids)
        if {n['id'] for n in active_nodes}-{n['id'] for n in nodes}:
            raise ValueError('Work selection lies outside the prepared execution scope')
        active_nodes=[n for n in active_nodes if n['stage']<=through_stage]
        if assigned is not None:
            mine=set(assigned)
            active_nodes=[n for n in active_nodes if n['source_case_id'] in mine]
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
        _atomic_json(root / f'latest-execution{suffix}.json', {'requested_workers': nproc,
            'active_worker_limit': workers, 'options': options, 'max_new_cases': max_new_cases, 'through_stage': through_stage,
            'work_case_ids':None if work_case_ids is None else sorted(work_case_ids),
            'scheduled_case_count':len(active_nodes),'imported_original_case_ids':imported,
            'shard':None if shard is None else list(shard),'assigned_source_count':None if assigned is None else len(assigned)})
        started = time.monotonic()
        new_count = 0
        in_flight = {}
        max_in_flight = 0
        pool = None
        def progress(status, **extra):
            _atomic_json(root / f'progress{suffix}.json', {'status': status, 'total_cases': len(nodes),
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
                    task_options=options
                    if origin is not None and ready['id'] in plan['member_extension']['extended_case_ids']:
                        previous=origin.get(ready['id'])
                        if previous is not None:
                            task_options={**options,'original_completion_id':previous['completion_id']}
                        elif (Path(plan['member_extension']['origin_execution'])/'tasks'/ready['id']).exists():
                            raise ValueError('Resolve the original incomplete case before extending its members')
                    future = pool.submit(_task, (str(store.root.resolve()), task, build_stage_contexts,
                        str((root / 'tasks' / ready['id']).resolve()), task_options))
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
            if shard is not None:
                result['shard']=list(shard)
                result['assigned_source_count']=len(assigned)
            _atomic_json(root / f'completion{suffix}.json', result)
            progress(status)
            return result
        except BaseException as exc:
            progress('FAILED', error_type=type(exc).__name__, error=str(exc))
            if pool is not None:
                # Lightning's SIGTERM handler keeps a worker alive; escalate so
                # the coordinator exits and releases its GPU allocation.
                processes = [p for p in tuple((pool._processes or {}).values()) if p.is_alive()]
                for process in processes:
                    process.terminate()
                for process in processes:
                    process.join(10)
                    if process.is_alive():
                        process.kill()
                        process.join(10)
                pool.shutdown(wait=False, cancel_futures=True)
                pool = None
            raise
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)


def adopt_completed_cases(store, plan, output, source_execution):
    """Register verified completions from an earlier execution of the same plan.

    For moving to a new deployment snapshot (for example a sharded launcher)
    without retraining: both executions must share the plan and store. Each
    adopted case is re-verified through the source registry and re-published,
    with its parents first, in the target registry. Nothing is trained, and the
    source execution is only read; it must not have an active coordinator.
    """
    root, old = Path(output), Path(source_execution)
    mine = json.loads((root / 'training-plan.json').read_text())
    theirs = json.loads((old / 'training-plan.json').read_text())
    plan_id = sha(canonical(plan))
    if not (mine['plan_sha256'] == theirs['plan_sha256'] == plan_id):
        raise ValueError('Adoption requires the same frozen plan')
    if mine['store'] != theirs['store'] or Path(mine['store']) != store.root.resolve():
        raise ValueError('Adoption requires the same artifact store')
    manifest = execution_manifest(store, plan, case_ids=mine['case_ids'], device=mine['device'],
        export_batch_size=mine['export_batch_size'],
        execution_patches=tuple(mine.get('execution_policy', {}).get('patches', ())))
    with (old / '.worker.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Source execution has an active coordinator') from exc
        with _owned_run(root, manifest, True):
            source = CaseRegistry(store, plan, old / 'registry', resume=True)
            target = CaseRegistry(store, plan, root / 'registry', resume=True)
            source.use_receipts = target.use_receipts = False  # adoption re-verifies fully
            adopted, present = [], 0
            for node in sorted(selected_nodes(plan, mine['case_ids']), key=lambda n: n['stage']):
                if target.get(node['id']) is not None:
                    present += 1
                    continue
                value = source.get(node['id'])
                if value is None:
                    continue
                target.publish(node['id'], value['task'], value['completion_id'])
                if target.get(node['id'])['completion_id'] != value['completion_id']:
                    raise ValueError('Adopted completion differs from its source')
                adopted.append(node['id'])
            report = {'status': 'ADOPTED_VERIFIED_COMPLETIONS', 'source_execution': str(old.resolve()),
                      'plan_sha256': plan_id, 'adopted_case_ids': adopted,
                      'already_present_cases': present}
            _atomic_json(root / f'adoption-{sha(canonical(sorted(adopted)))[:16]}.json', report)
            return report
