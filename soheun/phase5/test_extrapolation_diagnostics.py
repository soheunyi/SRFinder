"""Internal bias diagnostics preserve legacy background pulls and store provenance."""
import argparse,json,subprocess,sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'phase5')]
from artifacts.development_diagnostics import count_and_shape
from artifacts.extrapolation_diagnostics import summarize_extrapolation
from artifacts.campaign_runtime import run_campaign,import_sources
from artifacts.training_store import TrainingStore,canonical,sha
from artifacts.stage_processes import _initialize
from test_campaign_runtime import fixture
from correct_systematic_error import get_histograms
from utils import get_quantiles_with_weights


def legacy_background():
    rng=np.random.default_rng(93);n=16000
    train=rng.normal(size=n);tw=rng.uniform(.2,2,n);tg=rng.normal(0,.3,n)
    score=rng.normal(size=n);labels=np.arange(n)%2==1;truth=labels&(np.arange(n)%7==0)
    w=rng.uniform(.2,2,n);w[labels&(np.arange(n)%17==0)]*=-.2
    g=rng.normal(0,.3,n)
    events=SimpleNamespace(is_3b=~labels,is_4b=labels,is_signal=truth,is_bg4b=labels&~truth,weights=w)
    bins=get_quantiles_with_weights(train,tw*np.exp(tg),np.linspace(0,1,65))
    hist=get_histograms(events,score,bins,np.exp(g))
    expected=(hist['bg4b']-hist['3b_rw'])/np.sqrt(hist['4b_sq']+hist['3b_rw_sq'])
    actual=count_and_shape(train,tw,tg,score,labels,w,g,signal_truth=truth,background_only=True)
    np.testing.assert_allclose(actual['pull'],expected,rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(actual['relative_count_error'],(w[~labels]*np.exp(g[~labels])).sum()/w[labels&~truth].sum()-1,atol=1e-12)
    assert actual['target']=='background_4b' and actual['physical_4b_total']==w[labels].sum()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);_initialize();legacy_background()
    plan=fixture(args.out)
    cr=next(n for n in plan['nodes'] if n['stage']==3 and n['member_count']==2)
    for node in plan['nodes']:node['axes']['signal']='HH4b'
    infinite=deepcopy(cr);infinite['axes']['eta']='inf';infinite['requires']=cr['requires'][:1]
    infinite['id']=sha(canonical(infinite['axes']));plan['nodes'].append(infinite)
    ids=[cr['id'],infinite['id']]
    plan['evaluation_bindings']=[{'id':'power_test_grid','required_case_ids':ids}]
    plan['pilot_scope']={'signal':'HH4b','epsilon':['0'],'eta':['2.0','inf'],'sr_fraction':'.2',
                        'tier_A_seeds':[cr['axes']['mother_seed']],'tier_B_seeds':[]}
    store=TrainingStore(args.out/'store');import_sources(store,plan);execution=args.out/'training'
    run_campaign(store,plan,execution,case_ids=ids,nproc=1,device='cpu',export_batch_size=128)
    before=store.storage_stats();output=args.out/'internal'
    command=[sys.executable,str(ROOT/'phase5/diagnose_extrapolation_bias.py'),
        '--execution',str(execution),'--output',str(output),'--scope','pilot-A','--device','cpu','--batch-size','128']
    subprocess.run(command,check=True)
    summary=json.loads((output/'summary.json').read_text())
    assert summary['coverage']=='PARTIAL_DIAGNOSTIC_COVERAGE' and summary['placement']=='INTERNAL_NOT_MANUSCRIPT'
    assert len(summary['rows'])==8 and len(summary['pull_profiles'])==8
    assert all(r['n']==1 and r['target']=='all_4b' for r in summary['rows'])
    files={p.name:sha(p.read_bytes()) for p in output.iterdir() if p.suffix in ('.csv','.pdf')}
    assert len(files)==5 and all((output/name).stat().st_size>0 for name in files)
    subprocess.run(command+['--resume'],check=True)
    assert files=={name:sha((output/name).read_bytes()) for name in files}
    assert store.storage_stats()==before
    record=json.loads((output/'cases'/(cr['id']+'.json')).read_text())['value']
    full=[]
    for eta in ('2.0','inf'):
        for seed in range(100):
            value=deepcopy(record);value['axes'].update(eta=eta,mother_seed=seed);value['case_id']=f'{eta}-{seed}';full.append(value)
    assert summarize_extrapolation(full)['coverage']=='FULL_100_SEEDS_PER_CELL'
    try:summarize_extrapolation(full+[full[0]])
    except ValueError:pass
    else:raise AssertionError('Duplicate seed accepted')
    (args.out/'report.json').write_text(json.dumps({'status':'PASS','legacy_background_pull_and_count':True,
        'paired_store_backed_cli':True,'verified_resume_without_store_writes':True,
        'partial_coverage_visible':True,'four_internal_rule_figures':True})+'\n')
    print('PASS: internal background extrapolation diagnostics, legacy pull_bg4b, paired CLI, plots and immutable resume')

if __name__=='__main__':main()
