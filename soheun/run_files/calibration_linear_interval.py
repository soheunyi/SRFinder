import argparse
import json
import os
import pickle
from calibration_linear_band import REPO,one

OUT=REPO/'data/refit_bootstrap/linear_interval_eta1_sr020_v1'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--shards',type=int,default=20); ap.add_argument('--limit',type=int)
    args=ap.parse_args()
    source=REPO/'data/refit_bootstrap/noise_sweep_logitcap10_v1/manifest.pkl'
    rows=[r for r in pickle.load(open(source,'rb')) if r['noise_scale']==1. and r['sr_size']==.2]
    assert len(rows)==500
    rows=sorted(rows,key=lambda r:(r['signal_ratio'],r['seed']))
    (OUT/'results').mkdir(parents=True,exist_ok=True)
    selected=rows[args.shard::args.shards]
    if args.limit: selected=selected[:args.limit]
    for row in selected:
        dest=OUT/'results'/f"{row['hash']}.pkl"
        if dest.exists(): continue
        result=one(row,intervals=64)
        temp=dest.with_suffix(f'.{os.getpid()}.tmp')
        with open(temp,'wb') as f: pickle.dump(result,f)
        os.replace(temp,dest)
        print(json.dumps({k:result[k] for k in ('hash','signal_ratio','seed','p_value','global_band_p_value','seconds')}),flush=True)
if __name__=='__main__': main()
