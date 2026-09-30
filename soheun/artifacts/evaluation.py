"""Audited continuous-affine evaluation from completed immutable campaign artifacts."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib,importlib,json,time
from pathlib import Path
import numpy as np
from .training_store import TrainingStore,canonical,sha
from .case_registry import CaseRegistry
from .campaign_reader import CampaignReader
from .train_stage import _owned_run,_atomic_json

RULES=('single','mean_probability','mean_log_density_ratio','mean_density_ratio')
ROOT=Path(__file__).resolve().parents[1]


def open_reader(execution):
    execution=Path(execution)
    manifest=json.loads((execution/'training-plan.json').read_text())
    plan=json.loads((execution/'frozen-plan.json').read_text())
    if sha(canonical(plan))!=manifest['plan_sha256']:raise ValueError('Frozen plan checksum differs')
    store=TrainingStore(manifest['store'])
    return CampaignReader(CaseRegistry(store,plan,execution/'registry',resume=True)),manifest


def overlap_ranges(intervals,best):
    """Closed intervals attaining the reference's maximum overlap, as LD strings."""
    if not intervals:return [['0','1']]
    events={}
    for left,right in intervals:
        events.setdefault(left,[0,0])[0]+=1;events.setdefault(right,[0,0])[1]+=1
    points=sorted(events);current=0;segments=[]
    for i,x in enumerate(points):
        starts,ends=events[x];current+=starts
        if current==best:segments.append((x,x))
        current-=ends
        if current==best and i+1<len(points):segments.append((x,points[i+1]))
    merged=[]
    for left,right in segments:
        if merged and left<=merged[-1][1]:merged[-1]=(merged[-1][0],max(right,merged[-1][1]))
        else:merged.append((left,right))
    if not merged:raise ValueError('Reference maximum overlap could not be reconstructed')
    return [[str(x),str(y)] for x,y in merged]


@contextmanager
def capture_overlap(engine,signed):
    reference=engine.signed.ref if signed else engine.ref
    original=reference._max_overlap;detail={}
    def recorded(intervals):
        result=original(intervals)
        detail['maximizing_p_intervals']=overlap_ranges(intervals,result[0])
        return result
    reference._max_overlap=recorded
    try:yield detail
    finally:reference._max_overlap=original


def observed_at(arrays,lower,upper,t):
    z3,w3,z4,w4=arrays
    orders=[np.argsort(z,kind='stable') for z in (z3,z4)]
    z3,w3=z3[orders[0]],w3[orders[0]];z4,w4=z4[orders[1]],w4[orders[1]]
    position=(z3-lower)/(upper-lower)
    plus=w3*position;minus=w3*(1-position)
    plus/=plus.sum();minus/=minus.sum();r=w4/w4.sum()
    points=np.unique(np.r_[z3,z4]);i3=np.searchsorted(z3,points,side='right');i4=np.searchsorted(z4,points,side='right')
    def cdf(w,idx):
        c=np.r_[0.,np.cumsum(w,dtype=np.float64)];c[-1]=1.;return c[idx]
    return float(np.max(np.abs(t*cdf(plus,i3)+(1-t)*cdf(minus,i3)-cdf(r,i4))))


def checked_spec(plan):
    from phase5.evaluation_spec import verified_spec
    supplied=plan['evaluation_spec'];checked=verified_spec(supplied)
    if supplied.get('input_adapter_sha256')!=checked['input_adapter_sha256']:
        raise ValueError('Input adapter differs from the frozen evaluation specification')
    if checked['rng']['algorithm']!='numpy.random.default_rng (PCG64)':raise ValueError('Unsupported RNG algorithm')
    if not isinstance(np.random.default_rng().bit_generator,np.random.PCG64):raise ValueError('NumPy RNG default changed')
    return checked


def evaluate_one(reader,case_id,rule,spec):
    if rule not in RULES:raise ValueError('Unknown explicit aggregation rule')
    seed=spec['aggregation']['fixed_single_member_seed']
    if rule=='single' and type(seed) is not int:raise ValueError('Fixed single-member seed is undecided')
    started=time.perf_counter()
    arrays,audit=reader.affine_inputs(case_id,aggregation='mean_probability' if rule=='single' else rule,
        member_seeds=[seed] if rule=='single' else None,upper=spec['clipping']['upper'])
    signed=bool(audit['negative_4b_count'])
    module='run_files.affine_weighted_ks_signed_compiled' if signed else 'run_files.affine_weighted_ks_compiled'
    engine=importlib.import_module(module)
    loaded=time.perf_counter()
    with capture_overlap(engine,signed) as overlap:
        result=engine.affine_ks_test(*arrays,L=audit['lower'],U=audit['upper'],
            B=spec['bootstrap_replicates'],alpha=spec['alpha'],seed=spec['rng']['seed'],
            numerical_tol=spec['numerical_tolerance'])
    finished=time.perf_counter()
    record={'case_id':case_id,'axes':reader.registry.nodes[case_id]['axes'],'rule':rule,
            'test_version':spec['test_versions']['signed_4b' if signed else 'nonnegative'],
            'implementation_variant':'signed_4b' if signed else 'nonnegative',
            'result':asdict(result),'input_audit':audit,**overlap,
            'D_at_maximizing_p_t':observed_at(arrays,audit['lower'],audit['upper'],result.maximizing_p_t),
            'load_s':loaded-started,'bootstrap_s':finished-loaded}
    return record


def validate_record(record,identity,reader):
    if record['identity']!=identity or record['checksum']!=sha(canonical(record['value'])):
        raise ValueError('Evaluation record identity or checksum differs')
    value=record['value'];case=reader.case(value['case_id'])
    if value['input_audit']['stage_completion_id']!=case['completion_id']:
        raise ValueError('Evaluation upstream completion changed')
    return value


def run_evaluation(execution,output,case_ids,*,decision,rules=RULES,resume=False):
    reader,training_manifest=open_reader(execution);plan=reader.registry.plan;spec=checked_spec(plan)
    ids=sorted(case_ids);rules=list(rules)
    if not ids or len(set(ids))!=len(ids) or set(ids)-set(training_manifest['case_ids']):
        raise ValueError('Evaluation needs distinct cases from the prepared execution scope')
    if any(reader.registry.nodes[key]['stage']!=3 for key in ids):raise ValueError('Only completed CR cases can be tested')
    if not rules or len(set(rules))!=len(rules) or set(rules)-set(RULES):raise ValueError('Invalid comparison rules')
    if decision.get('primary_rule') not in RULES or not decision.get('decision_reference'):
        raise ValueError('Record the user decision and primary rule before power evaluation')
    binary=ROOT/'run_files/affine_envelope_kernel.so'
    binary_hash=hashlib.sha256(binary.read_bytes()).hexdigest()
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest={'schema':1,'kind':'continuous_affine_evaluation','training_execution':str(Path(execution).resolve()),
              'training_manifest_sha256':sha(canonical(training_manifest)),
              'plan_sha256':training_manifest['plan_sha256'],'spec':spec,'case_ids':ids,'rules':rules,
              'decision':decision,'numpy_version':np.__version__,'kernel_binary_sha256':binary_hash,
              'evaluation_source_sha256':source_hash}
    root=Path(output);finished=[]
    with _owned_run(root,manifest,resume):
        results=root/'results';results.mkdir(exist_ok=True)
        for key in ids:
            case=reader.case(key)
            for rule in rules:
                identity={'case_id':key,'completion_id':case['completion_id'],'rule':rule,
                          'evaluation_manifest_sha256':sha(canonical(manifest))}
                result_id=sha(canonical(identity));path=results/(result_id+'.json')
                if path.exists():validate_record(json.loads(path.read_text()),identity,reader)
                else:
                    value=evaluate_one(reader,key,rule,spec)
                    _atomic_json(path,{'identity':identity,'value':value,'checksum':sha(canonical(value))})
                finished.append(result_id)
                _atomic_json(root/'progress.json',{'status':'RUNNING','completed':len(finished),'expected':len(ids)*len(rules)})
        result={'status':'EVALUATION_COMPLETE','evaluation_manifest_sha256':sha(canonical(manifest)),
                'result_ids':finished,'case_count':len(ids),'rules':rules}
        _atomic_json(root/'completion.json',result)
        return result
