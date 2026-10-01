"""Regression checks for the independent review, including frozen legacy pull math."""
import json,sys,tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from artifacts.development_diagnostics import count_and_shape,decision_guidance,summarize_diagnostics
from artifacts.evaluation_summary import manuscript_table_rows
from artifacts.evaluation import validate_decision,resolve_decision
from artifacts.campaign_runtime import execution_manifest
from artifacts.training_store import TrainingStore
from correct_systematic_error import get_histograms
from utils import get_quantiles_with_weights


def rejects(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Invalid readiness input accepted')


def legacy_pulls():
    rng=np.random.default_rng(81);n=12000
    train=rng.normal(size=n);tw=rng.uniform(.2,3,n);tg=rng.normal(0,.3,n)
    score=rng.normal(size=n);labels=np.arange(n)%2==1;w=rng.uniform(.2,3,n)
    w[labels & (np.arange(n)%19==0)]*=-.15
    gamma=rng.normal(0,.3,n)
    # Use the actual old histogram function, train quantile helper and pull formula
    # from save_corrected_pulls_new.py / figure_scripts/null_case.py.
    events=SimpleNamespace(is_3b=~labels,is_4b=labels,is_bg4b=labels,weights=w,
                           is_signal=np.zeros(n,dtype=bool))
    bins=get_quantiles_with_weights(train,tw*np.exp(tg),np.linspace(0,1,65))
    hist=get_histograms(events,score,bins,np.exp(gamma))
    expected=(hist['4b']-hist['3b_rw'])/np.sqrt(hist['4b_sq']+hist['3b_rw_sq'])
    actual=count_and_shape(train,tw,tg,score,labels,w,gamma)
    assert actual['zero_variance_bins']==0
    np.testing.assert_allclose(actual['bin_edges'],bins,rtol=0,atol=1e-10)
    np.testing.assert_allclose(actual['pull'],expected,rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(actual['relative_count_error'],
        (w[~labels]*np.exp(gamma[~labels])).sum()/w[labels].sum()-1,rtol=0,atol=1e-12)


def guidance_coverage():
    cases=[]
    for eta in ('2.0','inf'):
        for seed in range(100):
            errors=np.array([np.sin(seed+i) for i in range(5)])
            rules={'single':{'relative_count_error':errors[0],'shape_error':1.}}
            for rule in ('mean_probability','mean_log_density_ratio','mean_density_ratio'):
                rules[rule]={'relative_count_error':float(errors.mean()),'shape_error':1.}
            cases.append({'axes':{'eta':eta,'mother_seed':seed},'member_seeds':list(range(5)),
                'member_metrics':{str(i):{'relative_count_error':float(e),'shape_error':1.} for i,e in enumerate(errors)},
                'rule_metrics':rules})
    tier_a=summarize_diagnostics([c for c in cases if c['axes']['mother_seed']<10])
    guidance=decision_guidance(tier_a)
    assert guidance['status']=='INSUFFICIENT_N' and 'indicative_rule' not in guidance
    full=summarize_diagnostics(cases)
    assert decision_guidance(full)['status']=='ADVISORY_ONLY'
    full['correlations']['inf']['mother_seeds'][-1]=100
    assert decision_guidance(full)['status']=='INSUFFICIENT_N'


def tables_and_decisions():
    rows=[{'eta':str(eta),'sr_fraction':str(sr),'epsilon':str(e),'rejection_rate':.25}
          for eta in (.5,1.,2.,3.) for sr in (.05,.1,.15,.2) for e in (0.,.005,.0075,.01,.02)]
    text=manuscript_table_rows(rows)
    assert text.count('\\multirow{4}')==4 and text.count('\\midrule')==3
    assert text.splitlines()[0]=='\\multirow{4}{*}{0.5} & $0.05$ & $0.25$ & $0.25$ & $0.25$ & $0.25$ & $0.25$ \\\\'
    rejects(lambda:manuscript_table_rows(rows[:-1]))
    decision={'status':'USER_DECISION_RECORDED','primary_rule':'mean_probability',
              'decision_reference':'synthetic fixture, not a scientific choice'}
    spec={'aggregation':{'status':'UNDECIDED','fixed_single_member_seed':0}}
    resolved=resolve_decision(spec,decision)
    assert spec['aggregation']['status']=='UNDECIDED'
    assert resolved['aggregation']['decision_sha256']==validate_decision(decision)
    rejects(lambda:validate_decision({k:v for k,v in decision.items() if k!='status'}))
    rejects(lambda:validate_decision({**decision,'status':'UNDECIDED'}))
    rejects(lambda:validate_decision({**decision,'decision_reference':' '}))
    with tempfile.TemporaryDirectory() as tmp:
        store=TrainingStore(Path(tmp)/'store')
        plan={'nodes':[{'id':'fixture','stage':1,'requires':[]}],
              'fixed_settings':{'torch_version':'intentionally-wrong','adam_epsilon':1e-8}}
        rejects(lambda:execution_manifest(store,plan))
        del plan['fixed_settings']['torch_version'];plan['fixed_settings']['adam_epsilon']=1e-4
        rejects(lambda:execution_manifest(store,plan))


if __name__=='__main__':
    legacy_pulls();guidance_coverage();tables_and_decisions()
    print('PASS: frozen legacy pull/count regression, strict 100-seed guidance, pivoted table, recorded decisions and runtime refusal')
