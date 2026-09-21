import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta
from calibration_continuous_affine import REPO,OUT,VERSION,BOOTSTRAPS

ap=argparse.ArgumentParser(); ap.add_argument('--require-complete',action='store_true'); args=ap.parse_args()
expected={r['hash']:r for r in pickle.load(open(OUT/'manifest.pkl','rb'))}
reference_hash=hashlib.sha256((REPO/'run_files/affine_weighted_ks_reference.py').read_bytes()).hexdigest()
adapter_hash=hashlib.sha256((REPO/'run_files/affine_weighted_ks_compiled.py').read_bytes()).hexdigest()
rows=[]
for path in (OUT/'results').glob('*.pkl'):
    r=pickle.load(open(path,'rb')); h=path.stem
    assert h in expected and r['hash']==h and r['version']==VERSION
    assert r['reference_sha256']==reference_hash and r['adapter_sha256']==adapter_hash
    assert r['bootstrap_replicates']==BOOTSTRAPS and r['statistic_clip']==10.
    assert r['alpha']==.05 and r['numerical_tolerance']==1e-12
    for key in ('signal_ratio','seed','sr_size','noise_scale'):
        assert r[key]==expected[h][key]
    assert r['p_value']==(1+r['max_exceedances'])/(BOOTSTRAPS+1)
    assert r['reject']==(r['p_value']<=.05)
    assert 0<=r['ks_t']<=1 and 0<=r['maximizing_p_t']<=1
    assert np.isfinite(r['ks_statistic']) and 0<=r['ks_statistic']<=1+1e-12
    rows.append({k:r[k] for k in ['hash','signal_ratio','seed','p_value','reject','ks_statistic','ks_t','maximizing_p_t','interval_band_p_value','global_band_p_value','load_seconds','bootstrap_seconds','n_clipped_3b','n_clipped_4b']})
assert len({r['hash'] for r in rows})==len(rows)
if args.require_complete: assert len(rows)==500
if not rows: print('No completed results'); raise SystemExit(0)
d=pd.DataFrame(rows)
baseline=pd.read_csv(REPO/'data/refit_bootstrap/noise_sweep_logitcap10_v1/noise_sweep_detailed.csv')
baseline=baseline[(baseline.noise_scale==1.)&(baseline.SR_size==.2)]
d=d.merge(baseline[['signal_ratio','seed','new_reject']],on=['signal_ratio','seed'],validate='one_to_one')
d['interval_reject']=d.interval_band_p_value<=.05
d['global_reject']=d.global_band_p_value<=.05
summary=d.groupby('signal_ratio').agg(n=('seed','size'),current_rate=('new_reject','mean'),global_rate=('global_reject','mean'),interval_rate=('interval_reject','mean'),continuous_rate=('reject','mean'),continuous_rejections=('reject','sum'),mean_bootstrap_seconds=('bootstrap_seconds','mean'))
if args.require_complete: assert (summary.n==100).all()
summary['lower95']=[0 if k==0 else beta.ppf(.025,k,n-k+1) for n,k in zip(summary.n,summary.continuous_rejections)]
summary['upper95']=[1 if k==n else beta.ppf(.975,k+1,n-k) for n,k in zip(summary.n,summary.continuous_rejections)]
d.to_csv(OUT/'paired_results.csv',index=False); summary.to_csv(OUT/'summary.csv')
audit={'complete':len(d)==500,'completed':len(d),'expected':500,'version':VERSION,'reference_sha256':reference_hash,'adapter_sha256':adapter_hash,'all_available_records_valid':True}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print('Completed',len(d),'/500'); print(summary.to_string())
