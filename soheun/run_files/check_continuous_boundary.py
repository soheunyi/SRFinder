import pandas as pd
import argparse
import json
import pickle
import numpy as np
from calibration_continuous_affine import OUT
ap=argparse.ArgumentParser(); ap.add_argument('--align-multipliers',action='store_true'); args=ap.parse_args()
d=pd.read_csv(OUT/'paired_results.csv')
x=d[(d.signal_ratio==.0075)&(d.reject!=d.interval_reject)]
print(x[['hash','seed','p_value','interval_band_p_value','reject','interval_reject']].to_string(index=False))
if args.align_multipliers:
    from calibration_continuous_affine import REPO,arrays
    from calibration_linear_band import band_test
    manifest={r['hash']:r for r in pickle.load(open(OUT/'manifest.pkl','rb'))}
    records=[]
    for _,row in x.iterrows():
        data,old,_,_=arrays(manifest[row['hash']]); s3,w3,s4,w4=data
        # The supplied implementation draws multipliers AFTER sorting each
        # class. Give the old bound that same event order to pair the draws.
        i3=np.argsort(s3,kind='stable'); i4=np.argsort(s4,kind='stable')
        r=band_test(s3[i3],s4[i4],w3[i3],w4[i4],old['lower'],reps=1000,seed=1729,intervals=64)
        assert row.p_value<=r['p_value']+1e-12
        record={'hash':row['hash'],'continuous_p':row.p_value,'previous_unaligned_interval_p':row.interval_band_p_value,'aligned_interval_p':r['p_value']}
        records.append(record); print(record,flush=True)
    (OUT/'boundary_multiplier_check.json').write_text(json.dumps(records,indent=2)+'\n')
