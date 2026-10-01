"""Compute pre-power null diagnostics; never choose a rule or launch training."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from artifacts.evaluation import open_reader
from artifacts.campaign_scope import expected_cases
from artifacts.training_store import canonical,sha
from artifacts.train_stage import _owned_run,_atomic_json
from artifacts.development_diagnostics import diagnose_case,summarize_diagnostics,decision_guidance
from artifacts.runtime_policy import numerical_state,runtime_policy
from artifacts.output_files import atomic_csv


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execution',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--tier',choices=['A','B','all'],required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--batch-size',type=int,default=32768);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.batch_size<1:raise ValueError('Positive inference batch size required')
    reader,training=open_reader(args.execution);plan=reader.registry.plan
    ids=sorted(k for k in expected_cases(plan,'pilot-'+args.tier) if float(reader.registry.nodes[k]['axes']['epsilon'])==0)
    if not ids or set(ids)-set(training['case_ids']):raise ValueError('Null development cases are outside the prepared scope')
    files=('artifacts/development_diagnostics.py','artifacts/regions.py','artifacts/model_loading.py','artifacts/training_store.py','utils.py','phase5/diagnose_campaign_ensemble.py')
    with runtime_policy(args.device):
        manifest={'schema':1,'kind':'pre_power_null_diagnostics','training_manifest_sha256':sha(canonical(training)),
            'case_ids':ids,'tier':args.tier,'nbins':64,'batch_size':args.batch_size,'device':args.device,
            'torch_version':str(torch.__version__),'numerics':numerical_state(),
            'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}}
        with _owned_run(args.output,manifest,args.resume):
            directory=args.output/'cases';directory.mkdir(exist_ok=True);values=[]
            for key in ids:
                current=reader.case(key);path=directory/(key+'.json')
                if path.exists():
                    record=json.loads(path.read_text());value=record['value']
                    if record['checksum']!=sha(canonical(value)) or value['stage_completion_id']!=current['completion_id']:
                        raise ValueError('Diagnostic record or upstream completion changed')
                else:
                    value=diagnose_case(reader,key,device=args.device,batch_size=args.batch_size)
                    _atomic_json(path,{'value':value,'checksum':sha(canonical(value))})
                values.append(value)
                _atomic_json(args.output/'progress.json',{'completed':len(values),'expected':len(ids)})
            summary=summarize_diagnostics(values);summary['guidance']=decision_guidance(summary)
            summary['tier']=args.tier;summary['manifest_sha256']=sha(canonical(manifest))
            _atomic_json(args.output/'summary.json',summary)
            atomic_csv(args.output/'diagnostics.csv',summary['rows'])
            _atomic_json(args.output/'completion.json',{'status':'NULL_DIAGNOSTICS_COMPLETE_USER_DECISION_PENDING',
                'case_count':len(values),'summary_sha256':sha(canonical(summary))})
            print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
