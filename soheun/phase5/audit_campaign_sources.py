"""Read-only candidate lookup for the frozen draft source inventory.

This checks metadata matching and record existence, not raw-pool integrity or
seed reconstruction. It never generates datasets or starts training.
"""
import argparse
from collections import Counter,defaultdict
from decimal import Decimal,InvalidOperation
import hashlib,json,pathlib,pickle,sys

SIGNAL_FILES={'HH4b':'HH4b_picoAOD.h5','HH4b_400':'HH4b_400.h5',
              'ZH4b':'ZH4b_picoAOD_cleaned.h5'}


def key(hp):
    return (str(hp['signal_filename']),str(Decimal(str(hp['signal_ratio'])).normalize()),
            str(Decimal(str(hp['seed'])).normalize()),str(Decimal(str(hp['n_3b'])).normalize()),str(Decimal(str(hp['ratio_4b'])).normalize()))


def audit(plan,metadata,data_root):
    index=defaultdict(list);invalid=0
    for record,hp in metadata.items():
        try:index[key(hp)].append(record)
        except (KeyError,ValueError,TypeError,InvalidOperation):invalid+=1
    rows=[]
    for node in plan['nodes']:
        if node['stage']!=1:continue
        axes=node['axes']
        expected={'signal_filename':SIGNAL_FILES[axes['signal']],
                  'signal_ratio':float(axes['epsilon']),'seed':axes['mother_seed'],
                  'n_3b':1000000,'ratio_4b':.5}
        candidates=sorted(index.get(key(expected),[]))
        present=[record for record in candidates if (data_root/'MotherSamples'/record).is_file()]
        status='UNIQUE_CANDIDATE' if len(candidates)==1 and len(present)==1 else 'MISSING' if not candidates else 'MISSING_RECORD_FILE' if not present else 'AMBIGUOUS'
        rows.append({'case_id':node['id'],'axes':axes,'expected_dataset':{**expected,'base_fvt_train_ratio':.5},
                     'status':status,'candidate_record_ids':candidates,'present_record_ids':present})
    return {'schema':1,'status':'SOURCE_CANDIDATE_AUDIT_NOT_VERIFIED','launch_submitted':False,
            'scope':'frozen draft Step-1 source cases only; metadata lookup and file existence',
            'counts':dict(Counter(row['status'] for row in rows)),'source_cases':len(rows),
            'metadata_records':len(metadata),'unindexed_metadata_records':invalid,'cases':rows,
            'remaining':['Verify source-record bytes, pool fingerprints, mother selections and outer splits before binding.',
                         'Resolve any ambiguous/missing candidates without guessing or adding source configurations.']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=pathlib.Path,required=True)
    ap.add_argument('--data-root',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--scan-unindexed',action='store_true')
    args=ap.parse_args();plan_bytes=args.plan.read_bytes()
    metadata_path=args.data_root/'metadata/MotherSamples.pkl';metadata_bytes=metadata_path.read_bytes()
    metadata=pickle.loads(metadata_bytes);original_count=len(metadata);scanned={}
    if args.scan_unindexed:
        sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
        from dataset import MotherSamples
        for path in sorted((args.data_root/'MotherSamples').iterdir()):
            if not path.is_file() or path.name in metadata:continue
            content=path.read_bytes();record=pickle.loads(content)
            if not isinstance(record,MotherSamples):raise ValueError(f'Unexpected source record: {path.name}')
            metadata[path.name]=record.hparams
            scanned[path.name]=hashlib.sha256(content).hexdigest()
            del record,content
    result=audit(json.loads(plan_bytes),metadata,args.data_root)
    result.update(original_metadata_records=original_count,scanned_unindexed_records=len(scanned),
                  scanned_record_sha256=scanned)

    result.update(plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps({key:result[key] for key in ('status','source_cases','counts','metadata_records','unindexed_metadata_records')}),flush=True)


if __name__=='__main__':main()
