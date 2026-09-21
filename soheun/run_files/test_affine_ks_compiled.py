import dataclasses
import json
import time
import numpy as np
import affine_weighted_ks_reference as slow
import affine_weighted_ks_compiled as fast

rng=np.random.default_rng(4132)
for case in range(500):
    n=int(rng.integers(1,300))
    s=rng.normal(size=n); a=rng.normal(size=n)
    if case%3==0: s=np.round(s)
    if case%7==0: s*=1e-12
    if case%11==0: a*=1e-12
    x=slow._upper_envelope(s,a); y=fast.upper_envelope(s,a)
    for key in ('left','right','slope','intercept'):
        assert np.array_equal(getattr(x,key),getattr(y,key)),(case,key)
for case in range(100):
    n3=int(rng.integers(10,100)); n4=int(rng.integers(10,100))
    x=rng.uniform(size=n3); y=rng.uniform(size=n4)
    if case%2==0: x=np.round(x,1); y=np.round(y,1)
    w=rng.lognormal(size=n3); r=rng.lognormal(size=n4)
    kwargs=dict(L=0.,U=1.,B=29,seed=case)
    a=slow.affine_ks_test(x,w,y,r,**kwargs)
    b=fast.affine_ks_test(x,w,y,r,**kwargs)
    assert dataclasses.asdict(a)==dataclasses.asdict(b),(case,a,b)
print('500 envelope and 100 complete-result EXACT equality checks passed',flush=True)
x=rng.uniform(size=100000); y=rng.uniform(size=100000)
for name,fn in [('reference',slow.affine_ks_test),('compiled',fast.affine_ks_test)]:
    start=time.perf_counter(); out=fn(x,np.ones(len(x)),y,np.ones(len(y)),L=0,U=1,B=20,seed=9)
    print(json.dumps({'implementation':name,'B':20,'n_per_class':len(x),'seconds':time.perf_counter()-start,'p':out.p_value}),flush=True)
