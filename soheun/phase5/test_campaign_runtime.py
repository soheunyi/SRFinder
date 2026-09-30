"""Actual small plan->recipes->spawn->receipt->registry chain, CPU by default.

Two synthetic sources share one pool. Tests do not represent scientific pilot
results and never submit Slurm or infer a production sweep.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'phase5')]
from constants import FEATURES
from dataset import SCDatasetInfo, MotherSamples
from artifacts.training_store import TrainingStore, canonical
from artifacts.bound_tasks import register_source_pointer
from artifacts.campaign_runtime import run_campaign, import_sources, selected_nodes, prepare_execution
from artifacts.case_registry import CaseRegistry
from artifacts.campaign_reader import CampaignReader
from artifacts.stage_processes import _task as stage_task
from test_three_stage_artifacts import hparams


def fixture(output):
    n = 2048
    torch.manual_seed(71)
    x = torch.rand(n, 4, 4)
    x[:, 0] = 40 + 100 * x[:, 0]
    x[:, 1] = 2 * x[:, 1] - 1
    x[:, 2] = 6 * x[:, 2] - 3
    x[:, 3] = 5 + 15 * x[:, 3]
    frame = pd.DataFrame(x.reshape(n, 16).numpy(), columns=FEATURES)
    frame['fourTag'] = np.random.RandomState(19).binomial(1, .4, n)
    frame['weight'] = 1 + np.arange(n, dtype=np.float64) / n
    pool = (output / 'pool.h5').resolve()
    frame.to_hdf(pool, key='df')
    origin = TrainingStore(output / 'source-store')
    plan = {'schema': 1, 'sources': {}, 'source_store': str(origin.root.resolve()), 'nodes': [],
            'templates': {}, 'status': 'SYNTHETIC_ENGINEERING_VALIDATION'}
    for role, stage in [('base', 1), ('smear', 2), ('raw_cr', 3), ('repr_cr', 2)]:
        hp = hparams(stage, 0)
        hp.pop('smearing')
        if role == 'repr_cr':
            hp.update(step=3, dim_q=6)
        plan['templates'][role] = hp
    for mother in (7, 8):
        params = {'seed': mother, 'n_3b': int((frame.fourTag == 0).sum()),
            'ratio_4b': float(frame.loc[frame.fourTag == 1, 'weight'].sum() / frame.weight.sum()),
            'signal_ratio': 0., 'signal_filename': pool.name, 'base_fvt_train_ratio': .5}
        source_hp = {**hparams(1, 0), 'dataset': params}
        raw = SCDatasetInfo([pool], [np.ones(n, dtype=bool)])
        mother_path = output / f'mother-{mother}.pkl'
        mother_path.write_bytes(pickle.dumps(MotherSamples(raw, 'fixture', params)))
        pointer = register_source_pointer(origin, mother_path, source_hp, source_root=output)
        ids = {stage: hashlib.sha256(f'{mother}-{stage}'.encode()).hexdigest() for stage in (1, 2, 3)}
        plan['sources'][ids[1]] = {'source': pointer}
        for stage in (1, 2, 3):
            plan['nodes'].append({'id': ids[stage], 'stage': stage, 'source_case_id': ids[1],
                'member_count': 2, 'member_seeds': [0, 1], 'max_epochs': 2,
                'axes': {'mother_seed': mother, 'epsilon': '0', 'eta': '2.0', 'sr_fraction': '.2',
                         'model': 'AttentionClassifier' if stage == 2 else 'FvTClassifier'},
                'requires': [] if stage == 1 else [ids[1]] if stage == 2 else [ids[1], ids[2]],
                'region_member_seeds': [0, 1]})
        plan['nodes'].append({'id': hashlib.sha256(f'{mother}-repr'.encode()).hexdigest(), 'stage': 3,
            'source_case_id': ids[1], 'member_count': 1, 'member_seeds': [1], 'max_epochs': 2,
            'axes': {'mother_seed': mother, 'epsilon': '0', 'eta': '2.0', 'sr_fraction': '.05',
                     'model': 'AttentionClassifier'}, 'requires': [ids[1], ids[2]],
            'region_member_seeds': [1], 'input_space': 'base_encoder', 'purpose': 'original_vs_representation'})
    for node in plan['nodes']:
        if node['stage']==3:node['region_recipe']='sr_quantile_cr_complement_v2'
    plan['nodes'].sort(key=lambda node: node['stage'])
    return plan


def expect_error(fn):
    try:
        fn()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError('Invalid input was accepted')


def interrupted_task(payload):
    store_root, spec, factory, output, options = payload
    return stage_task((store_root, spec, factory, output,
                       {**options, 'stop_after_completed_epochs': 1}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    plan = fixture(args.out)
    store = TrainingStore(args.out / 'store')
    assert import_sources(store, plan) == 2
    assert import_sources(store, plan) == 2
    plan_path = args.out / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    unused_output = args.out / 'dry-run-output'
    unused_store = args.out / 'dry-run-store'
    dry = json.loads(subprocess.check_output([sys.executable, str(ROOT / 'phase5/campaign.py'),
        'prepare', '--dry-run', '--plan', str(plan_path), '--output', str(unused_output),
        '--store', str(unused_store), '--device', args.device], text=True))
    assert dry['status'] == 'DRY_RUN_NO_WRITES_NO_SUBMISSION'
    assert not unused_output.exists() and not unused_store.exists()
    target = plan['nodes'][-1]['id']
    closure = selected_nodes(plan, [target])
    assert len(closure) == 3 and closure[-1]['id'] == target
    expect_error(lambda: selected_nodes(plan, ['missing']))
    duplicate = {**plan, 'nodes': plan['nodes'] + plan['nodes'][:1]}
    expect_error(lambda: selected_nodes(duplicate))
    options = {'device': args.device, 'export_batch_size': 128,
               'safety_bytes': 1024**3, 'compute_headroom_bytes': 1024**3}
    prepared = prepare_execution(store, plan, args.out / 'serial', device=args.device, export_batch_size=128)
    assert prepared['status'] == 'PREPARED_TRAINING_NOT_STARTED'
    assert not (args.out / 'serial' / 'tasks').exists()
    serial = run_campaign(store, plan, args.out / 'serial', nproc=1, resume=True, **options)
    assert serial['status'] == 'CAMPAIGN_ARTIFACTS_COMPLETE' and len(serial['completion_ids']) == 8
    status = json.loads(subprocess.check_output([sys.executable, str(ROOT / 'phase5/campaign.py'),
        'status', '--output', str(args.out / 'serial'), '--verify'], text=True))
    assert status['verified_cases'] == 8 and status['outputs_verified'] and status['runtime_matches']
    # A genuine completed-epoch interruption leaves recoverable state but no receipt.
    first_case = plan['nodes'][0]['id']
    with patch('artifacts.campaign_runtime._task', interrupted_task):
        try:
            run_campaign(store, plan, args.out / 'interrupted', case_ids=[first_case], nproc=1, **options)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Incomplete worker was registered as complete')
    assert not (args.out / 'interrupted' / 'completion.json').exists()
    recovery_checkpoint = args.out / 'interrupted' / 'tasks' / first_case / 'training/last.ckpt'
    assert recovery_checkpoint.is_file()
    recovered = run_campaign(store, plan, args.out / 'interrupted', case_ids=[first_case], nproc=1,
                             resume=True, **options)
    assert recovered['completion_ids'][first_case] == serial['completion_ids'][first_case]
    prefix = run_campaign(store, plan, args.out / 'parallel', nproc=2, through_stage=1, **options)
    assert set(prefix['completion_ids'])=={n['id'] for n in plan['nodes'] if n['stage']==1}
    assert prefix['status'] == 'CAMPAIGN_PREFIX_COMPLETE' and len(prefix['completion_ids']) == 2
    checkpoints = list((args.out / 'parallel' / 'tasks').glob('*/training/last.ckpt'))
    assert len(checkpoints) == 2
    before = {p: p.stat().st_mtime_ns for p in checkpoints}
    second = run_campaign(store, plan, args.out / 'parallel', nproc=1, resume=True, through_stage=2, **options)
    assert second['status']=='CAMPAIGN_PREFIX_COMPLETE'
    assert set(second['completion_ids'])=={n['id'] for n in plan['nodes'] if n['stage']<=2}
    parallel = run_campaign(store, plan, args.out / 'parallel', nproc=2, resume=True, through_stage=3, **options)
    assert parallel['max_in_flight'] <= 2
    assert parallel['completion_ids'] == serial['completion_ids']
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in before.items())
    all_checkpoints = list((args.out / 'parallel' / 'tasks').glob('*/training/last.ckpt'))
    all_stamps = {p: p.stat().st_mtime_ns for p in all_checkpoints}
    # Completed resume verifies outputs, then creates no process pool or refit.
    with patch('artifacts.campaign_runtime.ProcessPoolExecutor', side_effect=AssertionError('Unexpected refit')):
        resumed = run_campaign(store, plan, args.out / 'parallel', nproc=1, resume=True, **options)
    assert resumed['completion_ids'] == parallel['completion_ids']
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in all_stamps.items())
    reader_registry = CaseRegistry(store, plan, args.out / 'parallel' / 'registry', resume=True)
    reader = CampaignReader(reader_registry)
    cr_case = next(n['id'] for n in plan['nodes'] if n['stage'] == 3 and 'input_space' not in n)
    payload_count = store.storage_stats()['unique_payloads']
    pairs = reader.member_score_ids(cr_case, 'X2', member_seeds=[1, 0])
    assert [store.read(m, 'model')['identity']['training_recipe']['hparams']['model_seed'] for m, _ in pairs] == [1, 0]
    for reduction in ('mean_probability', 'mean_density_ratio', 'mean_log_density_ratio'):
        gamma, identity = reader.aggregate(cr_case, 'X2', aggregation=reduction)
        assert gamma.dtype == np.float32 and np.isfinite(gamma).all() and identity['aggregation'] == reduction
        arrays, audit = reader.affine_inputs(cr_case, aggregation=reduction, member_seeds=[0])
        assert audit['positive_normalizers'] and len(arrays[0]) == audit['n3']
    assert store.storage_stats()['unique_payloads'] == payload_count
    expect_error(lambda: reader.member_score_ids(cr_case, 'X1'))
    expect_error(lambda: reader.member_score_ids(cr_case, 'X2', member_seeds=[99]))
    masks = reader.held_out_regions(cr_case)
    assert np.all(masks['SR'] | masks['CR']) and not np.any(masks['SR'] & masks['CR'])
    assert not np.any(masks['neither'])
    bad = deepcopy(plan)
    bad['templates']['base']['optimizer']['lr'] = .001
    expect_error(lambda: run_campaign(store, bad, args.out / 'parallel', resume=True, **options))
    registry = CaseRegistry(store, plan, args.out / 'parallel' / 'registry', resume=True)
    case = plan['nodes'][0]['id']
    value = registry.get(case)
    wrong = deepcopy(value['task'])
    wrong['members'][0]['optimizer']['lr'] = .001
    expect_error(lambda: registry.publish(case, wrong, value['completion_id']))
    # A cached lookup may avoid full rereads only while all dependencies match.
    with patch.object(registry, '_verify', side_effect=AssertionError('Repeated verification')):
        assert registry.get(case) == value
    key = value['completion']['score_ids']['X2'][0]
    payload = store.payload_path(key)
    content = payload.read_bytes()
    payload.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    expect_error(lambda: registry.get(case))
    expect_error(lambda: CaseRegistry(store, plan, args.out / 'parallel' / 'registry', resume=True).get(case))
    payload.write_bytes(content)
    # The source digest cache must reject even a changed file with equal length.
    registry = CaseRegistry(store, plan, args.out / 'parallel' / 'registry', resume=True)
    registry.get(case)
    mother = Path(value['task']['source']['mother_record'])
    raw = mother.read_bytes()
    mother.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    expect_error(lambda: registry.get(case))
    mother.write_bytes(raw)
    report = {'status': 'PASS', 'device': args.device, 'logical_cases': 8,
        'serial_spawn_receipts_exact': True, 'staged_1_2_3_same_manifest_exact': True, 'prefix_resume_without_refit': True,
        'bounded_max_in_flight': parallel['max_in_flight'], 'completed_resume_without_workers': True,
        'frozen_optimizer_enforced': True, 'cached_dependency_mutation_rejected': True, 'epoch_interruption_recovery_exact': True,
        'cli_dry_run_no_writes': True, 'cli_status_fully_verified': True,
        'reader_seed_pairing_and_aggregation_explicit': True, 'derived_arrays_not_persisted': True}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
