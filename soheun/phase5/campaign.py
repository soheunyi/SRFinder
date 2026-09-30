"""Prepare, inspect and explicitly execute a frozen campaign on an allocation.

Preparation/dry-run never submit jobs. This CLI is an execution interface, not
an acceptance decision or permission to launch the manuscript campaign.
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from artifacts.training_store import TrainingStore, canonical, sha
from artifacts.campaign_runtime import (execution_manifest, prepare_execution,
                                        run_campaign, selected_nodes, runtime_snapshot)
from artifacts.case_registry import CaseRegistry


def gib(value):
    amount = float(value)
    if not math.isfinite(amount) or amount <= 0:
        raise argparse.ArgumentTypeError('A positive finite GiB amount is required')
    return int(amount * 1024**3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='operation', required=True)
    prep = sub.add_parser('prepare')
    for parser in (prep, sub.add_parser('run'), sub.add_parser('resume')):
        parser.add_argument('--plan', type=Path, required=True)
        parser.add_argument('--store', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        parser.add_argument('--case', action='append', dest='case_ids')
        parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
        parser.add_argument('--export-batch-size', type=int, default=1024)
        parser.add_argument('--execution-patches', default='', help='Optional comma-separated nosync,fast_gbn,fast_reinforce,graphs')
        if parser is prep:
            parser.add_argument('--dry-run', action='store_true')
            parser.add_argument('--resume', action='store_true')
        else:
            scope = parser.add_mutually_exclusive_group(required=True)
            scope.add_argument('--validation', action='store_true', help='A small explicitly selected engineering scope')
            scope.add_argument('--pilot-tier',choices=['A','B','all'],help='Run only the approved pilot tier within the full prepared scope')
            scope.add_argument('--start-campaign', action='store_true', help='User-authorized campaign execution')
            parser.add_argument('--nproc', type=int, default=5)
            parser.add_argument('--resident', choices=['auto', 'true', 'false'], default='auto')
            parser.add_argument('--safety-gib', type=gib)
            parser.add_argument('--compute-headroom-gib', type=gib)
            parser.add_argument('--max-new-cases', type=int)
            parser.add_argument('--through-stage',type=int,choices=[1,2,3],default=3,
                                help='Run only through this stage, preserving the full prepared scope')
    status = sub.add_parser('status')
    status.add_argument('--output', type=Path, required=True)
    status.add_argument('--verify', action='store_true', help='Fully verify registered case results and source inputs')
    cleanup = sub.add_parser('cleanup')
    cleanup.add_argument('--output',type=Path,required=True)
    cleanup.add_argument('--case',dest='case_ids',action='append',required=True)
    cleanup.add_argument('--apply',action='store_true',help='Prune verified completed working files; default is preview')
    args = ap.parse_args()
    if args.operation == 'cleanup':
        from artifacts.cleanup_completed import prune_completed_case
        manifest=json.loads((args.output/'training-plan.json').read_text())
        store=TrainingStore(manifest['store'])
        results=[prune_completed_case(store,args.output,key,apply=args.apply) for key in args.case_ids]
        print(json.dumps(results,indent=2),flush=True)
        return
    if args.operation == 'status':
        manifest = json.loads((args.output / 'training-plan.json').read_text())
        plan = json.loads((args.output / 'frozen-plan.json').read_text())
        if sha(canonical(plan)) != manifest['plan_sha256']:
            raise ValueError('Frozen plan checksum differs')
        report = {'plan_sha256': manifest['plan_sha256'], 'declared_cases': len(manifest['case_ids']),
                  'runtime_matches': runtime_snapshot() == manifest['runtime_files'], 'outputs_verified': False}
        progress = args.output / 'progress.json'
        report['progress'] = json.loads(progress.read_text()) if progress.exists() else {'status': 'PREPARED_TRAINING_NOT_STARTED'}
        if args.verify:
            store = TrainingStore(manifest['store'])
            registry_path = args.output / 'registry'
            if registry_path.exists():
                registry = CaseRegistry(store, plan, registry_path, resume=True)
                report['verified_cases'] = sum(registry.get(key) is not None for key in manifest['case_ids'])
            else: report['verified_cases'] = 0
            report['outputs_verified'] = True
        print(json.dumps(report, indent=2), flush=True)
        return
    if args.export_batch_size < 1:
        raise ValueError('Positive export batch size required')
    from speedups.policy import normalize
    patches = normalize(args.execution_patches.split(',') if args.execution_patches else [])
    plan = json.loads(args.plan.read_text())
    nodes = selected_nodes(plan, args.case_ids)
    if args.operation == 'prepare' and args.dry_run:
        # A shim prevents even creating store directories in this mode.
        manifest = execution_manifest(SimpleNamespace(root=args.store), plan, case_ids=args.case_ids,
            device=args.device, export_batch_size=args.export_batch_size, execution_patches=patches)
        print(json.dumps({'status': 'DRY_RUN_NO_WRITES_NO_SUBMISSION',
            'plan_sha256': manifest['plan_sha256'], 'runtime_sha256': manifest['runtime_sha256'],
            'cases': len(nodes), 'members_by_stage': {str(stage): sum(n['member_count'] for n in nodes if n['stage'] == stage)
                                                     for stage in (1, 2, 3)},
            'pending_gates': plan.get('gates', []), 'aggregation': plan.get('aggregation')}, indent=2), flush=True)
        return
    if args.operation != 'prepare' and args.validation:
        if not args.case_ids or len(nodes) > 30:
            raise ValueError('Validation requires an explicit scope of at most 30 dependency cases')
    store = TrainingStore(args.store)
    if args.operation == 'prepare':
        result = prepare_execution(store, plan, args.output, case_ids=args.case_ids, device=args.device,
            export_batch_size=args.export_batch_size, resume=args.resume, execution_patches=patches)
    else:
        # Explicit run after prepare is a resume of ownership, not necessarily a resumed fit.
        if not (args.output / 'prepared.json').is_file():
            raise ValueError('Prepare and review this execution scope before starting')
        resident = args.resident if args.resident == 'auto' else args.resident == 'true'
        work_cases=None
        if args.pilot_tier:
            from artifacts.campaign_scope import expected_cases
            work_cases=sorted(expected_cases(plan,'pilot-'+args.pilot_tier))
            if not work_cases:raise ValueError('Pilot selection is empty')
        result = run_campaign(store, plan, args.output, case_ids=args.case_ids, nproc=args.nproc,
            device=args.device, resident=resident, safety_bytes=args.safety_gib,
            compute_headroom_bytes=args.compute_headroom_gib, export_batch_size=args.export_batch_size,
            resume=True, max_new_cases=args.max_new_cases, execution_patches=patches, through_stage=args.through_stage, work_case_ids=work_cases)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
