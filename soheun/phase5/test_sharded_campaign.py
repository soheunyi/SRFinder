"""Equivalence gate for source-sharded (multi-GPU) campaign execution.

Concurrent shard coordinators on one prepared execution must reproduce the
single-coordinator completions exactly. Ownership locks must reject duplicate,
overlapping and mixed-mode coordinators, an interrupted shard must resume, and
adoption into a new execution must re-register verified completions without
training anything.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'phase5')]
from artifacts.training_store import TrainingStore
from artifacts.campaign_runtime import (run_campaign, import_sources, prepare_execution,
                                        adopt_completed_cases, shard_sources, selected_nodes)
from artifacts.case_registry import CaseRegistry
from test_campaign_runtime import fixture


def expect(error, fn, text):
    try:
        fn()
    except error as exc:
        if text not in str(exc):
            raise AssertionError(f'Unexpected message: {exc}')
        return
    raise AssertionError(f'Accepted; expected {error.__name__} mentioning {text!r}')


def final_json(text):
    """The CLI's result is the last top-level JSON object; Lightning logs precede it."""
    start = text.rfind('\n{')
    return json.loads(text[start + 1:] if start >= 0 else text)


def held(path, mode):
    handle = path.open('a+')
    fcntl.flock(handle, mode)
    return handle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    plan = fixture(args.out)
    plan_path = args.out / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    options = {'device': args.device, 'export_batch_size': 128,
               'safety_bytes': 1024**3, 'compute_headroom_bytes': 1024**3}

    # Reference: one unsharded coordinator on its own store.
    ref_store = TrainingStore(args.out / 'reference-store')
    import_sources(ref_store, plan)
    prepare_execution(ref_store, plan, args.out / 'reference', device=args.device, export_batch_size=128)
    reference = run_campaign(ref_store, plan, args.out / 'reference', nproc=1, resume=True, **options)
    assert reference['status'] == 'CAMPAIGN_ARTIFACTS_COMPLETE' and len(reference['completion_ids']) == 8

    # Assignment: whole sources, disjoint and covering; invalid specifications fail.
    nodes = selected_nodes(plan)
    first, second = shard_sources(nodes, (0, 2)), shard_sources(nodes, (1, 2))
    assert len(first) == len(second) == 1 and not set(first) & set(second)
    assert set(first) | set(second) == {node['source_case_id'] for node in nodes}
    expect(ValueError, lambda: shard_sources(nodes, (2, 2)), 'Shard must be')
    expect(ValueError, lambda: shard_sources(nodes, (0, 1)), 'Shard must be')
    crossed = [dict(node) for node in nodes]
    crossed[-1] = {**crossed[-1], 'requires': [n['id'] for n in nodes if n['source_case_id'] != crossed[-1]['source_case_id']][:1]}
    expect(ValueError, lambda: shard_sources(crossed, (0, 2)), 'Cross-source')

    # Sharded execution of the same plan on a fresh store.
    store = TrainingStore(args.out / 'store')
    import_sources(store, plan)
    root = args.out / 'sharded'
    expect(ValueError, lambda: run_campaign(store, plan, args.out / 'unprepared', shard=(0, 2), resume=True, **options),
           'prepared execution')
    prepare_execution(store, plan, root, device=args.device, export_batch_size=128)

    # Lock semantics; each lock below stands in for another live coordinator.
    with held(root / '.worker.lock', fcntl.LOCK_EX):
        expect(RuntimeError, lambda: run_campaign(store, plan, root, shard=(0, 2), resume=True, **options),
               'unsharded coordinator')
    (root / 'claims').mkdir(exist_ok=True)
    with held(root / 'claims' / 'shard-0-of-2.lock', fcntl.LOCK_EX):
        expect(RuntimeError, lambda: run_campaign(store, plan, root, shard=(0, 2), resume=True, **options),
               'already has an active coordinator')
    with held(root / 'claims' / f'source-{first[0]}.lock', fcntl.LOCK_EX):
        expect(RuntimeError, lambda: run_campaign(store, plan, root, shard=(0, 2), resume=True, **options),
               'claimed by another coordinator')
    with held(root / '.worker.lock', fcntl.LOCK_SH):
        expect(RuntimeError, lambda: run_campaign(store, plan, root, resume=True, **options), 'active worker')

    # Interrupted shard: one completed case, then a clean stop.
    prefix = run_campaign(store, plan, root, nproc=1, resume=True, shard=(0, 2), max_new_cases=1, **options)
    assert prefix['status'] == 'CAMPAIGN_PREFIX_COMPLETE' and prefix['shard'] == [0, 2]
    assert len(prefix['completion_ids']) == 1 and not prefix['selected_scope_complete']
    assert (root / 'progress-shard-0-of-2.json').is_file() and not (root / 'progress.json').exists()

    # Both shards concurrently, as separate CLI processes; shard 0 resumes.
    command = [sys.executable, str(ROOT / 'phase5/campaign.py'), 'resume', '--plan', str(plan_path),
               '--store', str(store.root), '--output', str(root), '--device', args.device,
               '--export-batch-size', '128', '--nproc', '1', '--safety-gib', '1', '--compute-headroom-gib', '1',
               '--validation', *[part for node in nodes for part in ('--case', node['id'])]]
    workers = [subprocess.Popen(command + ['--shard', f'{k}/2'], stdout=subprocess.PIPE, text=True)
               for k in (0, 1)]
    results = []
    for worker in workers:
        output, _ = worker.communicate()
        if worker.returncode:
            raise AssertionError(f'Shard coordinator failed with {worker.returncode}')
        results.append(final_json(output))
    assert all(result['selected_scope_complete'] for result in results)
    assert sorted(result['assigned_source_count'] for result in results) == [1, 1]
    registry = CaseRegistry(store, plan, root / 'registry', resume=True)
    union = {node['id']: registry.get(node['id'])['completion_id'] for node in nodes}
    assert union == reference['completion_ids'], 'Sharded completions differ from the single coordinator'

    # A completed shard trains nothing on rerun.
    again = run_campaign(store, plan, root, nproc=1, resume=True, shard=(1, 2), **options)
    assert again['selected_scope_complete'] and again['completion_ids'] == reference['completion_ids']

    # Status aggregates shard progress.
    status = json.loads(subprocess.check_output([sys.executable, str(ROOT / 'phase5/campaign.py'), 'status',
                                                 '--output', str(root)], text=True))
    assert set(status['shard_progress']) == {'shard-0-of-2', 'shard-1-of-2'}

    # Adoption into a new execution of the same plan and store: nothing trained.
    target = args.out / 'adopted'
    prepare_execution(store, plan, target, device=args.device, export_batch_size=128)
    with held(root / '.worker.lock', fcntl.LOCK_SH):
        expect(RuntimeError, lambda: adopt_completed_cases(store, plan, target, root), 'active coordinator')
    adopted = adopt_completed_cases(store, plan, target, root)
    assert len(adopted['adopted_case_ids']) == 8 and adopted['already_present_cases'] == 0
    repeat = adopt_completed_cases(store, plan, target, root)
    assert repeat['adopted_case_ids'] == [] and repeat['already_present_cases'] == 8
    resumed = run_campaign(store, plan, target, nproc=1, resume=True, **options)
    assert resumed['completion_ids'] == reference['completion_ids']
    assert not any((target / 'tasks').iterdir()), 'Adopted execution trained a case'
    expect(ValueError, lambda: adopt_completed_cases(ref_store, plan, target, root), 'same artifact store')

    print(json.dumps({'status': 'PASS', 'sharded_equals_single': True, 'concurrent_cli_shards': 2,
                      'interrupted_shard_resumed': True, 'duplicate_overlap_mixed_mode_rejected': True,
                      'unprepared_rejected': True, 'cross_source_dependency_rejected': True,
                      'shard_status_aggregated': True, 'adoption_without_training': True,
                      'adoption_idempotent': True, 'adoption_refuses_active_source': True}), flush=True)


if __name__ == '__main__':
    main()
