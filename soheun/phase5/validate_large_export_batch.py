"""Validate all native required-domain exports at a larger FP32 inference batch."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from artifacts.training_store import TrainingStore
from artifacts.case_registry import CaseRegistry
from artifacts.bound_tasks import resolve_source_pointer
from artifacts.export_stage import export_stage
from artifacts.stage_completion import complete_stage,verify_stage_completion
from artifacts.regions import define_regions,classify_X2
from artifacts.runtime_policy import runtime_policy


def main():
    ap=argparse.ArgumentParser()
    for name in ('plan','execution','store','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--batch-size',type=int,default=32768)
    args=ap.parse_args()
    if args.batch_size<1:raise ValueError('Positive batch size required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True,exist_ok=False)
    original=TrainingStore(args.store);overlay=TrainingStore(args.out/'store')
    # Immutable hardlinks avoid duplicating source arrays and best weights.
    for src,dst in ((original.records,overlay.records),(original.blobs,overlay.blobs)):
        for path in src.iterdir():
            if path.is_file() and not path.name.startswith('.pending-'):os.link(path,dst/path.name)
    plan=json.loads(args.plan.read_text())
    registry=CaseRegistry(original,plan,args.execution/'registry',resume=True)
    ids=json.loads((args.execution/'training-plan.json').read_text())['case_ids']
    cases=[registry.get(key) for key in ids]
    if any(c is None for c in cases):raise ValueError('Native chain incomplete')
    source=resolve_source_pointer(original,cases[0]['task']['source'])
    mapped={'X1':{},'X2':{}}
    report={'status':'RUNNING','batch_size':args.batch_size,'models':0,'score_arrays':0,'stages':[],'regions':[]}
    def save(): (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with runtime_policy('cuda'):
        for case in cases:
            r=case['completion']
            if r['dataset_id']!=source.dataset_id:raise ValueError('Mixed source')
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            out=export_stage(overlay,r['model_ids'],source,device='cuda',batch_size=args.batch_size,reuse_existing=False)
            torch.cuda.synchronize();wall=time.perf_counter()-start;comparisons=[]
            for domain,keys in out['score_ids'].items():
                for model,old,new in zip(r['model_ids'],r['score_ids'][domain],keys):
                    a=original.load_array(old,'scores');b=overlay.load_array(new,'scores')
                    exact=np.array_equal(a,b)
                    row={'model_id':model,'domain':domain,'exact':exact,'max_abs_logratio':float(np.abs(a-b).max())}
                    comparisons.append(row)
                    if not exact:
                        report.update(status='FAILED_PREDICTION_AGREEMENT',failure=row);save()
                        raise ValueError('Full-domain prediction changed')
                    mapped[domain][model]=new
            training={**r,'status':'TRAINING_COMPLETE_EXPORT_PENDING'}
            receipt=complete_stage(overlay,training,source,out['evaluation_splits'],out['score_ids'],out['event_metadata_ids'])
            verify_stage_completion(overlay,receipt,source)
            row={'case_id':case['task']['logical_case_id'],'stage':r['stage'],'models':len(r['model_ids']),
                 'score_arrays':len(comparisons),'export_wall_s':wall,'new_completion_id':receipt,
                 'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'comparisons':comparisons}
            report['stages'].append(row);report['models']+=row['models'];report['score_arrays']+=len(comparisons);save()
            print(json.dumps({k:v for k,v in row.items() if k!='comparisons'}),flush=True)
    for case in cases:
        if case['completion']['stage']!=3:continue
        u=case['task']['upstream'];old=original.read(u['region_id'],'region')
        def remap(keys,domain):
            return [mapped[domain][original.read(key,'scores')['identity']['owner_id']] for key in keys]
        base=remap(old['identity']['base_score_ids'],'X1')
        smooth=remap(old['identity']['smeared_score_ids'],'X1') or None
        new=define_regions(overlay,base,old['identity']['event_metadata_id'],smooth,
            quantile_recipe=old['identity']['quantile_recipe'],
            sr_fraction=old['identity']['requested_sr_fraction'],cr_fraction=old['identity']['requested_cr_fraction'],
            ensemble_mode=old['identity']['definition']['ensemble_mode'])
        if overlay.read(new,'region')['payload']!=old['payload']:raise ValueError('X1 thresholds changed')
        a=classify_X2(original,u['region_id'],u['base_X2_scores'],u.get('smeared_X2_scores'))
        b=classify_X2(overlay,new,remap(u['base_X2_scores'],'X2'),remap(u.get('smeared_X2_scores',[]),'X2') or None)
        if any(not np.array_equal(a[k],b[k]) for k in ('log_psi','SR','CR','neither')):
            raise ValueError('X2 region classification changed')
        report['regions'].append({'case_id':case['task']['logical_case_id'],'X1_thresholds_exact':True,
            'X2_logpsi_and_membership_exact':True,'upstream_members':len(base)})
    report.update(status='PASS_EXACT_ALL_NATIVE_DOMAINS',training_modified=False,
        reference_store_modified=False,total_export_wall_s=sum(r['export_wall_s'] for r in report['stages']),
        overlay_storage=overlay.storage_stats(),
        limitations=['One native source/configuration, not every family/device',
            'One export per case with current filesystem caches; not training speedup',
            'No production default is changed by this validation'])
    save();print(json.dumps({k:v for k,v in report.items() if k not in ('stages','regions')}),flush=True)

if __name__=='__main__':main()
