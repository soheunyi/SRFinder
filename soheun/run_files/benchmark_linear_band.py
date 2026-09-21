import pickle
import time
from calibration_linear_band import OUT,band_test
from recompute_null_rejection_rates import load_arrays

first=pickle.load(open(OUT/'results/250331_033453_480319_iboHES.pkl','rb'))
_,s3,s4,w3,w4,*_=load_arrays(first['hash'])
for batch in (1,4,8,16):
    start=time.perf_counter()
    r=band_test(s3,s4,w3,w4,first['lower'],reps=100,batch_size=batch)
    print('batch',batch,'seconds',time.perf_counter()-start,flush=True)
