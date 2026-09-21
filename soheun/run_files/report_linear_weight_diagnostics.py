import pickle
from pathlib import Path
import numpy as np
repo=Path(__file__).resolve().parents[1]
for name in ('linear_band_eta1_sr020_v1','linear_interval_eta1_sr020_v1'):
    null=[]
    for p in (repo/'data/refit_bootstrap'/name/'results').glob('*.pkl'):
        r=pickle.load(open(p,'rb'))
        if r['signal_ratio']==0: null.append(r)
    if not null: continue
    print(name,'null_n',len(null))
    for k in ('ess3','ess4','endpoint_ess','max_weight_share','lower','mixture_t'):
        x=np.array([r[k] for r in null])
        print(k,'min',x.min(axis=0),'median',np.median(x,axis=0),'max',x.max(axis=0))
    print('clipped_null_events',sum(r['n_clipped_3b']+r['n_clipped_4b'] for r in null))
