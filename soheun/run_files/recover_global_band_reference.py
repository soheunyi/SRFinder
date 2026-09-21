"""Recover missing global-band outputs from identical draws in the interval run."""
import os
import pickle
from pathlib import Path
import numpy as np

repo=Path(__file__).resolve().parents[1]
root=repo/'data/refit_bootstrap'
source=root/'linear_interval_eta1_sr020_v1/results'
destination=root/'linear_band_eta1_sr020_v1/results'
manifest=pickle.load(open(root/'noise_sweep_logitcap10_v1/manifest.pkl','rb'))
expected={r['hash'] for r in manifest if r['noise_scale']==1. and r['sr_size']==.2}
paths=list(source.glob('*.pkl'))
assert len(expected)==500 and {p.stem for p in paths}==expected
saved=0
for path in paths:
    r=pickle.load(open(path,'rb'))
    assert r['version']=='linear-interval-poisson-band-v1'
    dest=destination/path.name
    if dest.exists():
        old=pickle.load(open(dest,'rb'))
        assert np.allclose(old['bootstrap_values'],r['bootstrap_values'],atol=1e-12,rtol=0)
        assert np.isclose(old['p_value'],r['global_band_p_value'])
        continue
    r['p_value']=r.pop('global_band_p_value')
    for key in ('interval_p_values','interval_distances','intervals'):
        r.pop(key,None)
    r['version']='linear-simultaneous-poisson-band-v1'
    r['derived_from']=str(path)
    r['timing_note']='Runtime includes calculation of interval refinement in source run.'
    temp=dest.with_suffix(f'.{os.getpid()}.tmp')
    with open(temp,'wb') as f: pickle.dump(r,f)
    os.replace(temp,dest); saved+=1
print('Recovered identical global reference results:',saved)
