import json
import numpy as np
from calibration_linear_band import band_test

rng=np.random.default_rng(82341)
count=0; power=0
for seed in range(100):
    x=rng.uniform(size=500); w=np.exp(2*x)
    selected=[]
    while sum(map(len,selected))<500:
        z=rng.uniform(size=3000)
        rate=np.exp(2*z)*(.2+1.6*z)/(np.exp(2)*1.8)
        selected.append(z[rng.uniform(size=3000)<rate])
    y=np.concatenate(selected)[:500]
    r=band_test(x,y,w,np.ones(500),0.,1.,reps=199,seed=seed,intervals=64)
    assert r['p_value']<=r['global_band_p_value']
    count+=r['p_value']<=.05
    if seed<10:
        alt=np.clip(rng.normal(.5,.03,500),0,1)
        a=band_test(x,alt,w,np.ones(500),0.,1.,reps=199,seed=seed,intervals=64)
        power+=a['p_value']<=.05
print(json.dumps({'weighted_affine_null_rejections':count,'null_reps':100,'nonlinear_signal_rejections':power,'signal_reps':10}))
