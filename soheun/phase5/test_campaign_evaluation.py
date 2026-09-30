"""New-store evaluation must reproduce both frozen kernels and resume immutably."""
import argparse,json,sys,subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'phase5')]
from artifacts.evaluation import run_evaluation,open_reader,evaluate_one,overlap_ranges,RULES
from artifacts.training_store import TrainingStore
from artifacts.evaluation_summary import summarize,load_results
from artifacts.development_diagnostics import diagnose_case,summarize_diagnostics,count_and_shape
from artifacts.campaign_runtime import run_campaign,import_sources
from artifacts.affine_inputs import prepare_affine_inputs
from artifacts.stage_processes import _initialize
from evaluation_spec import verified_spec
from test_campaign_runtime import fixture
from test_speedup_integration import equal
from run_files.affine_weighted_ks_reference import affine_ks_test as reference
from run_files.affine_weighted_ks_signed_reference import affine_ks_test as signed_reference


def rejects(fn):
    try:fn()
    except (ValueError,FileNotFoundError):return
    raise AssertionError('Invalid evaluation input accepted')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);_initialize()
    plan=fixture(args.out);spec=verified_spec(json.loads((ROOT/'phase5/evaluation_spec.json').read_text()))
    plan['evaluation_spec']=spec
    store=TrainingStore(args.out/'store');import_sources(store,plan)
    case=next(n['id'] for n in plan['nodes'] if n['stage']==3 and n['member_count']==2)
    for node in plan['nodes']:node['axes']['signal']='HH4b'
    plan['evaluation_bindings']=[{'id':'power_test_grid','required_case_ids':[case]}]
    plan['pilot_scope']={'signal':'HH4b','epsilon':['0'],'eta':['2.0','inf'],'sr_fraction':'.2','tier_A_seeds':[7,8],'tier_B_seeds':[]}
    execution=args.out/'training'
    run_campaign(store,plan,execution,case_ids=[case],nproc=1,device='cpu',export_batch_size=128)
    decision={'primary_rule':'mean_probability','decision_reference':'synthetic test fixture; not a scientific choice'}
    before=store.storage_stats()
    result=run_evaluation(execution,args.out/'evaluation',[case],decision=decision)
    assert result['status']=='EVALUATION_COMPLETE' and len(result['result_ids'])==4
    reader,_=open_reader(execution)
    diagnostic=diagnose_case(reader,case,device='cpu',batch_size=128,nbins=8)
    assert diagnostic['member_seeds']==[0,1] and set(diagnostic['rule_metrics'])==set(RULES)
    one=summarize_diagnostics([diagnostic]);assert len(one['rows'])==4
    assert all(r['predicted_K15_RMS'] is None for r in one['rows'])
    # Analytic zero-correlation errors: averaging two halves the variance.
    fake=[]
    for index,errors in enumerate(((1.,-1.),(-1.,1.),(1.,1.),(-1.,-1.))):
        metrics={str(i):{'relative_count_error':e,'shape_error':1.} for i,e in enumerate(errors)}
        fake.append({'axes':{'eta':'2.0','mother_seed':index},'member_seeds':[0,1],
            'member_metrics':metrics,'rule_metrics':{'mean_density_ratio':{'relative_count_error':sum(errors)/2,'shape_error':1.}}})
    projection=summarize_diagnostics(fake)
    assert projection['correlations']['2.0']['empirical_member_error_correlation']==0.
    assert np.isclose(projection['rows'][0]['predicted_K15_RMS'],np.sqrt(1/15))
    assert store.storage_stats()==before
    for p in (args.out/'evaluation/results').glob('*.json'):
        record=json.loads(p.read_text())['value'];rule=record['rule']
        arrays,audit=reader.affine_inputs(case,aggregation='mean_probability' if rule=='single' else rule,
            member_seeds=[0] if rule=='single' else None)
        expected=reference(*arrays,L=audit['lower'],U=10.,B=1000,alpha=.05,seed=1729,numerical_tol=1e-12)
        equal(record['result'],asdict(expected))
        assert record['maximizing_p_intervals'] and record['D_at_maximizing_p_t']>=0
    with patch('artifacts.evaluation.evaluate_one',side_effect=AssertionError('Cached bootstrap rerun')):
        assert run_evaluation(execution,args.out/'evaluation',[case],decision=decision,resume=True)==result
    rejects(lambda:run_evaluation(execution,args.out/'evaluation',[case],decision={**decision,'primary_rule':'single'},resume=True))
    rejects(lambda:run_evaluation(execution,args.out/'missing-decision',[case],decision={}))
    path=next((args.out/'evaluation/results').glob('*.json'));saved=path.read_bytes()
    value=json.loads(saved);value['value']['result']['p_value']=.123;path.write_text(json.dumps(value))
    try:rejects(lambda:run_evaluation(execution,args.out/'evaluation',[case],decision=decision,resume=True))
    finally:path.write_bytes(saved)
    actual=evaluate_one;calls=[]
    def interrupt(*args,**kwargs):
        calls.append(True)
        if len(calls)==2:raise RuntimeError('synthetic interruption')
        return actual(*args,**kwargs)
    with patch('artifacts.evaluation.evaluate_one',side_effect=interrupt):
        try:run_evaluation(execution,args.out/'interrupted',[case],decision=decision)
        except RuntimeError:pass
        else:raise AssertionError('Expected evaluation interruption')
    assert not (args.out/'interrupted/completion.json').exists()
    resumed=run_evaluation(execution,args.out/'interrupted',[case],decision=decision,resume=True)
    assert len(resumed['result_ids'])==4
    assert store.storage_stats()==before
    summary=summarize([args.out/'evaluation'],args.out/'summary',scope='full')
    assert summary['case_count']==1 and summary['cells']==4
    rejects(lambda:load_results([args.out/'evaluation',args.out/'evaluation']))
    rejects(lambda:summarize([args.out/'evaluation'],args.out/'wrong-rules',scope='full',rules=['single']))
    complete=args.out/'evaluation/completion.json';raw_complete=complete.read_bytes()
    edited=json.loads(raw_complete);edited['result_ids']=edited['result_ids'][:-1];complete.write_text(json.dumps(edited))
    try:rejects(lambda:load_results([args.out/'evaluation']))
    finally:complete.write_bytes(raw_complete)
    subprocess.run([sys.executable,str(ROOT/'phase5/render_campaign_power.py'),
        '--summary',str(args.out/'summary'),'--output',str(args.out/'figures')],check=True)
    assert (args.out/'figures/power_plot_HH4b_noise_scale=2.0.pdf').stat().st_size>0
    assert not (args.out/'wrong-rules').exists()
    subprocess.run([sys.executable,str(ROOT/'phase5/diagnose_campaign_ensemble.py'),
        '--execution',str(execution),'--output',str(args.out/'development'),
        '--tier','A','--device','cpu','--batch-size','128'],check=True,stdout=subprocess.DEVNULL)
    assert json.loads((args.out/'development/completion.json').read_text())['status']=='NULL_DIAGNOSTICS_COMPLETE_USER_DECISION_PENDING'
    subprocess.run([sys.executable,str(ROOT/'phase5/diagnose_campaign_ensemble.py'),
        '--execution',str(execution),'--output',str(args.out/'development'),
        '--tier','A','--device','cpu','--batch-size','128','--resume'],check=True,stdout=subprocess.DEVNULL)
    # Signed physical weights plus a score below -10 verify the frozen upper-only adapter.
    keys=np.column_stack((np.zeros(40,dtype=np.int64),np.arange(40)))
    labels=np.arange(40)%2==1;weights=np.ones(40);weights[1]=-.1
    arrays,audit=prepare_affine_inputs(keys,labels,weights,np.linspace(-15,15,40),np.zeros(40),
        dataset_id='signed-fixture',domain='X2',lower=-20,upper=10)
    assert min(arrays[0].min(),arrays[2].min())<-10 and max(arrays[0].max(),arrays[2].max())==10
    mock=SimpleNamespace(affine_inputs=lambda *a,**k:(arrays,audit),registry=SimpleNamespace(nodes={'signed':{'axes':{}}}))
    signed=evaluate_one(mock,'signed','single',spec)
    expected=signed_reference(*arrays,L=-20,U=10,B=1000,alpha=.05,seed=1729,numerical_tol=1e-12)
    equal(signed['result'],asdict(expected));assert signed['implementation_variant']=='signed_4b'
    assert overlap_ranges([(np.longdouble(0),np.longdouble('.5')),(np.longdouble('.5'),np.longdouble(1))],2)==[['0.5','0.5']]
    report={'status':'PASS','all_four_rules_match_reference':True,'signed_variant_matches_reference':True,
            'upper_only_clipping':True,'closed_interval_ties':True,'interrupted_resume':True,
            'cached_reuse_no_bootstrap':True,'corrupt_result_and_changed_decision_rejected':True,
            'permanent_training_store_unchanged':True,'complete_scope_summary_and_new_root_power_plot':True,
            'duplicate_shards_and_missing_results_rejected':True,'transient_X1_null_diagnostics_no_store_writes':True,
            'ensemble_variance_projection_analytic_case':True,'development_cli_and_resume':True}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
