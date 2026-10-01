"""Explicit optional member-count decision -> new plan/execution; never submit jobs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from artifacts.training_store import TrainingStore,canonical,sha
from artifacts.member_extension import make_extension_plan,origin_guard
from artifacts.campaign_runtime import prepare_execution,run_campaign
from artifacts.campaign_scope import expected_cases


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--origin',type=Path,required=True)
    ap.add_argument('--plan-out',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--decision',type=Path,required=True,help='Recorded user step3_member_count and decision_reference')
    ap.add_argument('--members',type=int,choices=[15],default=15)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--nproc',type=int,default=5)
    ap.add_argument('--export-batch-size',type=int,default=32768)
    ap.add_argument('--execution-patches',default='nosync,fast_gbn,fast_reinforce,graphs')
    ap.add_argument('--safety-gib',type=float,default=3);ap.add_argument('--compute-headroom-gib',type=float,default=5)
    ap.add_argument('--scope',choices=['pilot-A','pilot-B','pilot-all','full'],required=True)
    ap.add_argument('--execute',action='store_true',help='Run only after the recorded member-count decision; default is preparation')
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    if args.nproc<1 or args.export_batch_size<1 or args.safety_gib<0 or args.compute_headroom_gib<0:
        raise ValueError('Invalid worker, batch or memory settings')
    original=json.loads((args.origin/'frozen-plan.json').read_text())
    manifest=json.loads((args.origin/'training-plan.json').read_text())
    plan=make_extension_plan(original,manifest,args.origin,member_count=args.members,
                             decision=json.loads(args.decision.read_text()))
    store=TrainingStore(manifest['store'])
    with origin_guard(store,plan):pass
    if args.device!=manifest['device'] or args.export_batch_size!=manifest['export_batch_size']:
        raise ValueError('Extension must preserve the original device/export profile')
    args.plan_out.parent.mkdir(parents=True,exist_ok=True)
    if args.plan_out.exists():
        if not args.resume or json.loads(args.plan_out.read_text())!=plan:
            raise ValueError('Extension plan output already exists or differs')
    else:store._publish(args.plan_out,canonical(plan))
    from speedups.policy import normalize
    patches=normalize(args.execution_patches.split(',') if args.execution_patches else [])
    prepared=prepare_execution(store,plan,args.output,case_ids=manifest['case_ids'],device=args.device,
        export_batch_size=args.export_batch_size,resume=args.resume,execution_patches=patches)
    if not args.execute:
        print(json.dumps(prepared),flush=True);return
    work=sorted(expected_cases(plan,args.scope))
    result=run_campaign(store,plan,args.output,case_ids=manifest['case_ids'],work_case_ids=work,
        nproc=args.nproc,device=args.device,resident='auto',export_batch_size=args.export_batch_size,
        safety_bytes=int(args.safety_gib*1024**3),compute_headroom_bytes=int(args.compute_headroom_gib*1024**3),
        resume=True,execution_patches=patches)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
