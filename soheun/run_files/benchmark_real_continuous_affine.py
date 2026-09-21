from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import affine_weighted_ks_reference as reference
import affine_weighted_ks_compiled as compiled
from calibration_continuous_affine import arrays,targets,OUT

row=targets()[0]; data,old,_,_=arrays(row)
report={'hash':row['hash'],'n3':len(data[0]),'n4':len(data[2]),'B':20,'seed':1729}
outputs={}
for name,fn in [('reference',reference.affine_ks_test),('compiled',compiled.affine_ks_test)]:
    start=time.perf_counter()
    outputs[name]=asdict(fn(*data,L=old['lower'],U=10.,B=20,seed=1729))
    report[name+'_seconds']=time.perf_counter()-start
assert outputs['reference']==outputs['compiled']
report['all_result_fields_exactly_equal']=True
report['speedup']=report['reference_seconds']/report['compiled_seconds']
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'real_reference_equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
