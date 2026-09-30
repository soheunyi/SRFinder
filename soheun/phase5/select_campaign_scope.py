"""Write a declared evaluation/pilot case list; never train or submit jobs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from artifacts.campaign_scope import expected_cases
from artifacts.campaign_runtime import selected_nodes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--scope',choices=['full','pilot-A','pilot-B','pilot-all'],required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    plan=json.loads(args.plan.read_text());cases=sorted(expected_cases(plan,args.scope))
    if not cases:raise ValueError('Scope is empty')
    closure=selected_nodes(plan,cases)
    with args.output.open('x') as handle:json.dump(cases,handle,indent=2);handle.write('\n')
    print(json.dumps({'scope':args.scope,'test_cases':len(cases),'members_by_stage':{
        str(stage):sum(n['member_count'] for n in closure if n['stage']==stage) for stage in (1,2,3)}}),flush=True)

if __name__=='__main__':main()
