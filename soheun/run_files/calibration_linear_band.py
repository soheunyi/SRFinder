"""Focused diagnostic: simultaneous Poisson multiplier band for affine tilts.

Conditional asymptotic validity requires a true nonnegative affine correction,
independent event samples, negligible dominating weights, and fixed learned maps.
This is NOT an unconditional calibration theorem for a misspecified CR network.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / 'data/refit_bootstrap/linear_band_eta1_sr020_v1'


def band_test(s3, s4, w3, w4, lower, upper=10., reps=1000, seed=1729, batch_size=1, intervals=0):
    """One shared multiplier per physical event, joint across endpoint tilts."""
    s3, s4, w3, w4 = [np.asarray(x, dtype=np.float64) for x in (s3,s4,w3,w4)]
    for s,w in ((s3,w3),(s4,w4)):
        if not np.isfinite(s).all() or not np.isfinite(w).all() or np.any(w<0):
            raise ValueError('Nonfinite data or negative weights')
        if np.min(s)<lower-1e-7 or np.max(s)>upper+1e-7 or w.sum()<=0:
            raise ValueError('Invalid fixed support/weight total')
    # The cone of nonnegative affine functions on [lower,upper] is generated
    # by s-lower and upper-s. Its normalized CDFs form a line segment.
    a = w3 * np.maximum(s3-lower,0.)
    b = w3 * np.maximum(upper-s3,0.)
    if min(a.sum(),b.sum())<=0:
        raise ValueError('Degenerate affine endpoint')
    a /= a.sum(); b /= b.sum(); c = w4/w4.sum()
    support, inverse = np.unique(np.r_[s3,s4], return_inverse=True)
    n3=len(s3); g3=inverse[:n3]; g4=inverse[n3:]; m=len(support)
    def cumsum_group(values, groups):
        return np.cumsum(np.bincount(groups,weights=values,minlength=m))
    fa=cumsum_group(a,g3); fb=cumsum_group(b,g3); fc=cumsum_group(c,g4)
    def objective(t):
        return np.max(np.abs(t*fa+(1-t)*fb-fc))
    fit=minimize_scalar(objective,bounds=(0.,1.),method='bounded',options={'xatol':1e-12})
    t=min([0.,1.,float(fit.x)],key=objective)
    observed=float(objective(t))
    rng=np.random.default_rng(seed)
    boot=np.empty(reps)
    endpoint_a=np.empty(reps); endpoint_b=np.empty(reps)
    order=np.argsort(np.r_[s3,s4],kind='stable')
    ordered_scores=np.r_[s3,s4][order]
    ends=np.r_[np.flatnonzero(np.diff(ordered_scores)!=0),len(order)-1]
    wa=np.r_[a,np.zeros(len(s4))][order]
    wb=np.r_[b,np.zeros(len(s4))][order]
    wc=np.r_[np.zeros(n3),c][order]
    for start in range(0,reps,batch_size):
        stop=min(start+batch_size,reps)
        # Rows are independent replicates; draw in original event order to
        # make results exactly reproducible across different batch sizes.
        mult=(rng.poisson(1.,size=(stop-start,len(order)))-1.)[:,order]
        def process(weight,cdf):
            terms=mult*weight
            return np.cumsum(terms,axis=1)[:,ends]-terms.sum(axis=1)[:,None]*cdf
        zc=process(wc,fc)
        za=process(wa,fa)-zc
        ma=np.max(np.abs(za),axis=1)
        del za
        zb=process(wb,fb)-zc
        mb=np.max(np.abs(zb),axis=1)
        boot[start:stop]=np.maximum(ma,mb)
        endpoint_a[start:stop]=ma; endpoint_b[start:stop]=mb
    result=dict(observed=observed,mixture_t=t,
                p_value=float((1+np.count_nonzero(boot>=observed-1e-10))/(reps+1)),
                critical95=float(np.quantile(boot,.95,method='higher')),
                bootstrap_values=boot,lower=lower,upper=upper,
                ess3=float(w3.sum()**2/(w3@w3)),ess4=float(w4.sum()**2/(w4@w4)),
                endpoint_ess=[float(1/(a@a)),float(1/(b@b))],
                max_weight_share=[float(a.max()),float(b.max()),float(c.max())])
    if intervals:
        grid=np.linspace(0.,1.,intervals+1)
        # D(t) is convex; constrained minimum is at the global minimizer
        # projected onto the interval. No repeated optimization is needed.
        interval_t=np.clip(t,grid[:-1],grid[1:])
        distances=np.array([objective(x) for x in interval_t])
        pvals=[]
        for left,right,d in zip(grid[:-1],grid[1:],distances):
            bound=np.maximum(left*endpoint_a+(1-left)*endpoint_b,
                             right*endpoint_a+(1-right)*endpoint_b)
            pvals.append((1+np.count_nonzero(bound>=d-1e-10))/(reps+1))
        result.update(global_band_p_value=result['p_value'],
                      p_value=float(max(pvals)),interval_p_values=np.array(pvals),
                      intervals=intervals,interval_distances=distances)
    return result


def one(row, intervals=0):
    from run_files.recompute_null_rejection_rates import load_arrays
    from training_info import TrainingInfo
    from signal_region import compute_sr_stats,get_SR_CR_cut
    from dataset import MotherSamples
    from constants import FEATURES
    from events_data import events_from_scdinfo
    start=time.time()
    ti,s3,s4,w3,w4,nc3,nc4=load_arrays(row['hash'])
    cfg=ti.hparams['signal_region']; filename=ti.hparams['dataset']['signal_filename']
    train,_=compute_sr_stats(cfg['SR_stats_hashes'],filename,cfg['ensemble_mode'],cfg['stats_type'])
    source=TrainingInfo.load(cfg['SR_stats_hashes'][0]); mother=MotherSamples.load(source.ms_hash)
    events=events_from_scdinfo(mother.scdinfo[source.ms_idx],FEATURES,filename)
    cutoff,_=get_SR_CR_cut(train,events,{'4b_in_SR':cfg['4b_in_SR'],'4b_in_CR':cfg['4b_in_CR']})
    lower=float(np.clip(cutoff,-10,10))
    result=band_test(s3,s4,w3,w4,lower,intervals=intervals)
    version='linear-simultaneous-poisson-band-v1' if not intervals else 'linear-interval-poisson-band-v1'
    result.update(row,version=version,n_reps=1000,
                  n_clipped_3b=nc3,n_clipped_4b=nc4,seconds=time.time()-start)
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--shards',type=int,default=20); ap.add_argument('--limit',type=int)
    args=ap.parse_args()
    source=REPO/'data/refit_bootstrap/noise_sweep_logitcap10_v1/manifest.pkl'
    rows=[r for r in pickle.load(open(source,'rb')) if r['noise_scale']==1. and r['sr_size']==.2]
    assert len(rows)==500 and len({r['hash'] for r in rows})==500
    rows=sorted(rows,key=lambda r:(r['signal_ratio'],r['seed']))
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'results').mkdir(exist_ok=True)
    selected=rows[args.shard::args.shards]
    if args.limit: selected=selected[:args.limit]
    for row in selected:
        dest=OUT/'results'/f"{row['hash']}.pkl"
        if dest.exists(): continue
        result=one(row); temp=dest.with_suffix(f'.{os.getpid()}.tmp')
        with open(temp,'wb') as f: pickle.dump(result,f)
        os.replace(temp,dest)
        print(json.dumps({k:result[k] for k in ('hash','signal_ratio','seed','p_value','seconds')}),flush=True)

if __name__=='__main__': main()
