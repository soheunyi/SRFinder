import numpy as np
from calibration_linear_band import band_test

rng=np.random.default_rng(42)
# Identical weighted distributions, including ties: exact zero discrepancy.
s=np.array([0.,0.,.2,.7,1.,1.]); w=np.array([1.,2.,3.,2.,1.,2.])
r=band_test(s,s,w,w,0.,1.,reps=99)
assert r['observed']<1e-9 and r['p_value']==1.
# Invariance to independent overall rescaling of class weights.
s3=rng.uniform(size=200); s4=rng.uniform(size=220)
w3=rng.uniform(.5,2.,200); w4=rng.uniform(.5,2.,220)
r1=band_test(s3,s4,w3,w4,0.,1.,reps=99)
r2=band_test(s3,s4,17*w3,.03*w4,0.,1.,reps=99)
r_batch1=band_test(s3,s4,w3,w4,0.,1.,reps=99,batch_size=1)
r_batch8=band_test(s3,s4,w3,w4,0.,1.,reps=99,batch_size=8)
assert np.allclose(r_batch8['bootstrap_values'],r_batch1['bootstrap_values'],atol=1e-14)
localized=band_test(s3,s4,w3,w4,0.,1.,reps=99,intervals=64)
assert localized['p_value']<=r1['p_value']
assert np.all(localized['interval_distances']>=r1['observed']-1e-10)
assert np.allclose(r1['bootstrap_values'],r2['bootstrap_values'],atol=1e-14)
assert abs(r1['observed']-r2['observed'])<1e-10
# Reference bootstrap fluctuations evaluated directly at every unique score.
grid=np.unique(np.r_[s3,s4]); a=w3*s3; a/=a.sum(); b=w3*(1-s3); b/=b.sum(); c=w4/w4.sum()
gen=np.random.default_rng(1729); u=gen.poisson(1.,len(s3))-1; v=gen.poisson(1.,len(s4))-1
f3=(s3[:,None]<=grid); f4=(s4[:,None]<=grid)
za=(u*a)@(f3-a@f3); zb=(u*b)@(f3-b@f3); zc=(v*c)@(f4-c@f4)
reference=max(np.max(abs(za-zc)),np.max(abs(zb-zc)))
assert np.isclose(reference,r1['bootstrap_values'][0],atol=1e-14)
null_reject=0; alt_reject=0
for j in range(100):
    x=rng.uniform(size=400)
    # True affine tilt 2*x has triangular CDF x**2.
    y=np.sqrt(rng.uniform(size=400))
    rr=band_test(x,y,np.ones(400),np.ones(400),0.,1.,reps=199,seed=j)
    null_reject+=rr['p_value']<=.05
    if j<10:
        y=np.clip(rng.normal(.5,.03,400),0,1)
        aa=band_test(x,y,np.ones(400),np.ones(400),0.,1.,reps=199,seed=j)
        alt_reject+=aa['p_value']<=.05
print('checks_passed; synthetic_affine_null_rejects',null_reject,'/100; nonlinear_signal_rejects',alt_reject,'/10')
