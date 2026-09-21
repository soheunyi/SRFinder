"""Audit frozen target coverage and saved numerical results, without reruns."""
import json
import pickle
from pathlib import Path
import numpy as np

repo=Path(__file__).resolve().parents[1]
manifest=pickle.load(open(repo/'data/refit_bootstrap/noise_sweep_logitcap10_v1/manifest.pkl','rb'))
expected={r['hash']:r for r in manifest if r['noise_scale']==1. and r['sr_size']==.2}
assert len(expected)==500
records={}
for name,version in [('linear_band_eta1_sr020_v1','linear-simultaneous-poisson-band-v1'),('linear_interval_eta1_sr020_v1','linear-interval-poisson-band-v1')]:
    base=repo/'data/refit_bootstrap'/name
    found={}
    for path in (base/'results').glob('*.pkl'):
        r=pickle.load(open(path,'rb')); h=path.stem
        assert h in expected and r['hash']==h and r['version']==version
        for key in ('noise_scale','sr_size','signal_ratio','seed'):
            assert r[key]==expected[h][key],(h,key)
        assert r['n_reps']==1000 and r['upper']==10.
        assert -10<=r['lower']<10 and 0<=r['mixture_t']<=1
        assert np.isfinite(r['bootstrap_values']).all() and len(r['bootstrap_values'])==1000
        assert np.isfinite(r['observed']) and r['observed']>=0
        global_p=(1+np.count_nonzero(r['bootstrap_values']>=r['observed']-1e-10))/1001
        if 'interval_p_values' in r:
            assert r['intervals']==64 and len(r['interval_p_values'])==64
            assert np.isclose(r['global_band_p_value'],global_p)
            assert r['p_value']==max(r['interval_p_values'])
            assert r['p_value']<=global_p+1e-12
        else:
            assert np.isclose(r['p_value'],global_p)
        found[h]=r
    counts={str(e):sum(r['signal_ratio']==e for r in found.values()) for e in (0.,.005,.0075,.01,.02)}
    report={'complete':len(found)==500,'completed':len(found),'expected':500,'counts':counts,'missing':sorted(expected.keys()-found.keys())}
    if base.exists():
        (base/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(name,len(found),counts,'all_available_results_valid')
    records[name]=found
common=records['linear_band_eta1_sr020_v1'].keys() & records['linear_interval_eta1_sr020_v1'].keys()
for h in common:
    a=records['linear_band_eta1_sr020_v1'][h]; b=records['linear_interval_eta1_sr020_v1'][h]
    assert np.isclose(a['observed'],b['observed'],atol=1e-12,rtol=0)
    assert np.allclose(a['bootstrap_values'],b['bootstrap_values'],atol=1e-12,rtol=0)
print('paired_method_input_and_draw_agreement',len(common))
