"""Explicit store-backed bootstrap evaluation; no training or Slurm submission."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from artifacts.evaluation import run_evaluation,RULES


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execution',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    choice=ap.add_mutually_exclusive_group(required=True)
    choice.add_argument('--case',action='append',dest='cases')
    choice.add_argument('--case-file',type=Path,help='JSON list of declared case IDs')
    ap.add_argument('--decision',type=Path,required=True,help='JSON status=USER_DECISION_RECORDED, primary_rule and user decision_reference')
    ap.add_argument('--rules',nargs='+',choices=RULES,default=list(RULES))
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    cases=json.loads(args.case_file.read_text()) if args.case_file else args.cases
    decision=json.loads(args.decision.read_text())
    result=run_evaluation(args.execution,args.output,cases,decision=decision,rules=args.rules,resume=args.resume)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
