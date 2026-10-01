"""Summarize checked evaluation shards into a new output root."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from artifacts.evaluation_summary import summarize
from artifacts.evaluation import RULES


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evaluation',type=Path,action='append',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--scope',choices=['full','pilot-A','pilot-B','pilot-all'],required=True)
    ap.add_argument('--rules',nargs='+',choices=RULES,default=list(RULES))
    args=ap.parse_args();report=summarize(args.evaluation,args.output,scope=args.scope,rules=args.rules)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
