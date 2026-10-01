"""Derive internal HH4b eta=infinity/2 SR background diagnostics from best models."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from artifacts.evaluation import open_reader
from artifacts.campaign_scope import expected_cases
from artifacts.development_diagnostics import diagnose_case
from artifacts.extrapolation_diagnostics import summarize_extrapolation,write_extrapolation
from artifacts.training_store import canonical,sha
from artifacts.train_stage import _owned_run,_atomic_json
from artifacts.runtime_policy import runtime_policy,numerical_state


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execution',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--scope',choices=['full','pilot-A','pilot-B','pilot-all'],required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--batch-size',type=int,default=32768)
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    if args.batch_size<1:raise ValueError('Positive inference batch size required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    reader,training=open_reader(args.execution);nodes=reader.registry.nodes
    ids=sorted(k for k in expected_cases(reader.registry.plan,args.scope)
        if nodes[k]['axes']['signal']=='HH4b' and float(nodes[k]['axes']['eta']) in (2.,float('inf')))
    if not ids or set(ids)-set(training['case_ids']):raise ValueError('Internal diagnostic cases are outside execution scope')
    files=('artifacts/development_diagnostics.py','artifacts/extrapolation_diagnostics.py',
           'artifacts/regions.py','artifacts/model_loading.py','artifacts/training_store.py',
           'utils.py','phase5/diagnose_extrapolation_bias.py')
    with runtime_policy(args.device):
        manifest={'schema':1,'kind':'internal_extrapolation_diagnostics','scope':args.scope,
            'training_manifest_sha256':sha(canonical(training)),'case_ids':ids,'nbins':64,
            'device':args.device,'batch_size':args.batch_size,'torch_version':str(torch.__version__),
            'numerics':numerical_state(),'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}}
        with _owned_run(args.output,manifest,args.resume):
            directory=args.output/'cases';directory.mkdir(exist_ok=True);values=[]
            for key in ids:
                current=reader.case(key);path=directory/(key+'.json')
                if path.exists():
                    record=json.loads(path.read_text());value=record['value']
                    if (record['checksum']!=sha(canonical(value)) or value['case_id']!=key
                            or value['stage_completion_id']!=current['completion_id']):
                        raise ValueError('Internal diagnostic record or upstream changed')
                else:
                    value=diagnose_case(reader,key,device=args.device,batch_size=args.batch_size,
                                        nbins=64,analysis='extrapolation_bias')
                    _atomic_json(path,{'value':value,'checksum':sha(canonical(value))})
                values.append(value)
                _atomic_json(args.output/'progress.json',{'completed':len(values),'expected':len(ids)})
            summary=summarize_extrapolation(values);summary['manifest_sha256']=sha(canonical(manifest))
            _atomic_json(args.output/'summary.json',summary)
            if (args.output/'figure-provenance.json').exists():
                provenance=json.loads((args.output/'figure-provenance.json').read_text())
                if (provenance['summary_sha256']!=sha(canonical(summary)) or
                    any(sha((args.output/name).read_bytes())!=digest for name,digest in provenance['outputs'].items())):
                    raise ValueError('Internal diagnostic output changed')
            else:write_extrapolation(summary,args.output)
            _atomic_json(args.output/'completion.json',{'status':'INTERNAL_EXTRAPOLATION_DIAGNOSTICS_COMPLETE',
                'case_count':len(values),'coverage':summary['coverage'],'summary_sha256':sha(canonical(summary))})
            print(json.dumps({'status':'INTERNAL_EXTRAPOLATION_DIAGNOSTICS_COMPLETE',
                'case_count':len(values),'coverage':summary['coverage']}),flush=True)

if __name__=='__main__':main()
