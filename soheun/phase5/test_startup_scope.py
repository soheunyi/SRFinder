"""Coordinator startup verifies only its own work, and adoption can run in source shards.

1. A coordinator scheduling one Step-3 case fully verifies only that case and its
   transitive dependencies (the rest is listed from the registry index), yet reports
   every registered completion; with receipts present it fully verifies nothing.
2. Two concurrent `campaign.py adopt --shard K/2` processes migrate every case with
   identical completions; repeating adopts nothing; a live source shard claim blocks it.
"""
import argparse, fcntl, json, shutil, subprocess, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'phase5')]
from artifacts.training_store import TrainingStore
from artifacts.campaign_runtime import run_campaign, import_sources, prepare_execution, selected_nodes, adopt_completed_cases
import artifacts.case_registry as cr
from test_campaign_runtime import fixture
from test_sharded_campaign import final_json


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
    plan = fixture(args.out); plan_path = args.out / 'plan.json'; plan_path.write_text(json.dumps(plan))
    store = TrainingStore(args.out / 'store'); import_sources(store, plan)
    old = args.out / 'old'; opts = dict(device='cpu', export_batch_size=128)
    prepare_execution(store, plan, old, **opts)
    full = run_campaign(store, plan, old, nproc=1, resume=True, **opts)
    assert full['status'] == 'CAMPAIGN_ARTIFACTS_COMPLETE'
    nodes = {n['id']: n for n in selected_nodes(plan)}

    calls = []; original = cr.CaseRegistry._verify
    def counting(self, case_id, *a, **k):
        calls.append(case_id); return original(self, case_id, *a, **k)
    cr.CaseRegistry._verify = counting
    target = next(k for k, n in nodes.items() if n['stage'] == 3)
    deps = set(); pending = [target]
    while pending:
        k = pending.pop()
        if k not in deps: deps.add(k); pending.extend(nodes[k]['requires'])
    shutil.rmtree(old / 'verification-receipts')
    calls.clear()
    one = run_campaign(store, plan, old, nproc=1, resume=True, work_case_ids=[target], **opts)
    assert set(calls) == deps and len(deps) < len(nodes), (sorted(calls), sorted(deps))
    assert one['completion_ids'] == full['completion_ids'] and one['selected_scope_complete']
    calls.clear()
    run_campaign(store, plan, old, nproc=1, resume=True, work_case_ids=[target], **opts)
    assert not calls, calls
    cr.CaseRegistry._verify = original

    new = args.out / 'new'; prepare_execution(store, plan, new, **opts)
    claims = old / 'claims'; claims.mkdir(exist_ok=True)
    with (claims / 'shard-0-of-2.lock').open('a+') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        try:
            adopt_completed_cases(store, plan, new, old, shard=(0, 2))
        except RuntimeError as exc:
            assert 'active coordinator' in str(exc), exc
        else:
            raise AssertionError('Adoption ran beside a live source shard')
    cmd = [sys.executable, str(ROOT / 'phase5/campaign.py'), 'adopt', '--output', str(new), '--from', str(old)]
    procs = [subprocess.Popen(cmd + ['--shard', f'{k}/2'], stdout=subprocess.PIPE, text=True) for k in (0, 1)]
    results = []
    for p in procs:
        out, _ = p.communicate()
        assert p.returncode == 0, p.returncode
        results.append(final_json(out))
    adopted = sorted(k for r in results for k in r['adopted_case_ids'])
    assert adopted == sorted(nodes), (len(adopted), len(nodes))
    reg = cr.CaseRegistry(store, plan, new / 'registry', resume=True)
    assert {k: reg.get(k)['completion_id'] for k in nodes} == full['completion_ids']
    again = adopt_completed_cases(store, plan, new, old, shard=(1, 2))
    assert again['adopted_case_ids'] == [] and again['already_present_cases'] > 0
    print(json.dumps({'status': 'PASS', 'cases': len(nodes), 'one_case_full_verifications': len(deps),
                      'warm_full_verifications': 0, 'sharded_adoption_cases': len(adopted),
                      'adoption_blocked_by_live_source_shard': True}), flush=True)


if __name__ == '__main__':
    main()
