"""Synthetic importance-weighted affine null; the classes are not exchangeable."""
import numpy as np
from calibration_linear_band import band_test

rng=np.random.default_rng(9131)
rejections=0
for seed in range(100):
    x=rng.uniform(size=500)
    # 3b weighted distribution proportional to exp(2*x).
    w=np.exp(2*x)
    accepted=[]
    while sum(len(a) for a in accepted)<500:
        candidate=rng.uniform(size=3000)
        density=np.exp(2*candidate)*(.2+1.6*candidate)
        accepted.append(candidate[rng.uniform(size=3000)<density/(np.exp(2)*1.8)])
    y=np.concatenate(accepted)[:500]
    r=band_test(x,y,w,np.ones(500),0.,1.,reps=199,seed=seed)
    rejections+=r['p_value']<=.05
print('importance_weighted_affine_null_rejections',rejections,'/100')
