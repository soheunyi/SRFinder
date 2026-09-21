import pickle
import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import beta

repo=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(); ap.add_argument('--interval',action='store_true'); args=ap.parse_args()
name='linear_interval_eta1_sr020_v1' if args.interval else 'linear_band_eta1_sr020_v1'
version='linear-interval-poisson-band-v1' if args.interval else 'linear-simultaneous-poisson-band-v1'
base=repo/'data/refit_bootstrap'/name
rows=[]
for path in (base/'results').glob('*.pkl'):
    r=pickle.load(open(path,'rb'))
    assert r['version']==version
    rows.append({k:r[k] for k in ['hash','signal_ratio','seed','p_value','observed','critical95','seconds','mixture_t','ess3','ess4']})
if not rows:
    print('No results yet'); raise SystemExit(0)
d=pd.DataFrame(rows); assert not d.duplicated(['signal_ratio','seed']).any()
d['band_reject']=d.p_value<=.05
previous=pd.read_csv(repo/'data/refit_bootstrap/noise_sweep_logitcap10_v1/noise_sweep_detailed.csv')
previous=previous[(previous.noise_scale==1.)&(previous.SR_size==.2)]
d=d.merge(previous[['signal_ratio','seed','new_reject']],on=['signal_ratio','seed'],validate='one_to_one')
summary=d.groupby('signal_ratio').agg(n=('seed','size'),current_refit_rate=('new_reject','mean'),band_rate=('band_reject','mean'),band_rejections=('band_reject','sum'),mean_seconds=('seconds','mean'))
summary['lower95']=[0 if k==0 else beta.ppf(.025,k,n-k+1) for n,k in zip(summary.n,summary.band_rejections)]
summary['upper95']=[1 if k==n else beta.ppf(.975,k+1,n-k) for n,k in zip(summary.n,summary.band_rejections)]
print('completed',len(d),'/500'); print(summary.to_string())
d.to_csv(base/'paired_results.csv',index=False); summary.to_csv(base/'summary.csv')
