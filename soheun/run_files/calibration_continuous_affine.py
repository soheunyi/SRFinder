"""Focused evaluation of the supplied test, with an equivalent compiled hull."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import time
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
OUT=REPO/'data/refit_bootstrap/continuous_affine_eta1_sr020_v1'
SOURCE=REPO/'data/refit_bootstrap/noise_sweep_logitcap10_v1/manifest.pkl'
PREVIOUS=REPO/'data/refit_bootstrap/linear_interval_eta1_sr020_v1/results'
VERSION='continuous-affine-ks-poisson-v1'
BOOTSTRAPS=1000
SEED=1729

def targets():
    rows=[r for r in pickle.load(open(SOURCE,'rb')) if r['noise_scale']==1. and r['sr_size']==.2]
    assert len(rows)==500 and len({r['hash'] for r in rows})==500
    assert all(sum(r['signal_ratio']==e for r in rows)==100 for e in (0.,.005,.0075,.01,.02))
    return sorted(rows,key=lambda r:(r['signal_ratio'],r['seed']))

def freeze_manifest():
    rows=targets(); OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'manifest.pkl'
    if path.exists(): assert pickle.load(open(path,'rb'))==rows
    else:
        with open(path,'xb') as f: pickle.dump(rows,f)
    (OUT/'results').mkdir(exist_ok=True)
    print('frozen targets',len(rows),flush=True)

def arrays(row):
    from run_files.recompute_null_rejection_rates import load_arrays
    ti,s3,s4,w3,w4,nc3,nc4=load_arrays(row['hash'])
    old=pickle.load(open(PREVIOUS/f"{row['hash']}.pkl",'rb'))
    for key in ('hash','signal_ratio','seed','noise_scale','sr_size'):
        assert old[key]==row[key],(row['hash'],key)
    assert old['upper']==10. and old['version']=='linear-interval-poisson-band-v1'
    # Reuse the audited, training-defined SR cutoff; load_arrays already applies
    # original SR membership before clipping. No model or region is changed.
    return (s3,w3,s4,w4),old,nc3,nc4

def run_one(row):
    from affine_weighted_ks_compiled import affine_ks_test
    start=time.perf_counter(); data,old,nc3,nc4=arrays(row)
    loaded=time.perf_counter()
    r=affine_ks_test(*data,L=old['lower'],U=10.,B=BOOTSTRAPS,alpha=.05,seed=SEED)
    assert 0<=r.max_exceedances<=BOOTSTRAPS
    out=asdict(r)
    out.update(row,version=VERSION,statistic_clip=10.,lower=old['lower'],upper=10.,
               n_clipped_3b=nc3,n_clipped_4b=nc4,rng_seed=SEED,
               interval_band_p_value=old['p_value'],
               global_band_p_value=old['global_band_p_value'],
               load_seconds=loaded-start,bootstrap_seconds=time.perf_counter()-loaded)
    out['reference_sha256']=hashlib.sha256((REPO/'run_files/affine_weighted_ks_reference.py').read_bytes()).hexdigest()
    out['adapter_sha256']=hashlib.sha256((REPO/'run_files/affine_weighted_ks_compiled.py').read_bytes()).hexdigest()
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--shard',type=int,default=0); ap.add_argument('--shards',type=int,default=50)
    ap.add_argument('--limit',type=int)
    args=ap.parse_args()
    if args.prepare: freeze_manifest(); return
    rows=pickle.load(open(OUT/'manifest.pkl','rb'))
    selected=rows[args.shard::args.shards]
    if args.limit: selected=selected[:args.limit]
    for row in selected:
        dest=OUT/'results'/f"{row['hash']}.pkl"
        if dest.exists():
            old=pickle.load(open(dest,'rb'))
            assert old['version']==VERSION and old['bootstrap_replicates']==BOOTSTRAPS
            continue
        r=run_one(row); temp=dest.with_suffix(f'.{os.getpid()}.tmp')
        with open(temp,'wb') as f: pickle.dump(r,f)
        os.replace(temp,dest)
        print(json.dumps({k:r[k] for k in ('hash','signal_ratio','seed','p_value','interval_band_p_value','bootstrap_seconds')}),flush=True)

if __name__=='__main__': main()
