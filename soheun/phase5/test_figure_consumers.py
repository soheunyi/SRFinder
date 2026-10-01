"""Verify figure math/provenance and plotting guards without scientific claims."""
import argparse,json,sys,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from artifacts.training_store import TrainingStore,canonical,sha
from artifacts.figure_data import signal_efficiency,upstream_efficiency,model_aware_score
from plots import get_weights_by_sr_stats
from phase5.render_draft_efficiency import render
from artifacts.illustration_data import RegionOverlay,weighted_calibration,class_histograms,upstream_view
from render_campaign_illustrations import classifier,overlap,calibration as calibration_plot,tail,representation
from render_campaign_figures import copy_static


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    store=TrainingStore(args.out/'store');dataset=store.put_dataset(sha(b'figure-fixture'),8,{'fixture':True})
    split=store.put_split(dataset,'X2',np.arange(8,dtype=np.int64),{'algorithm':'fixture','seed':1,'version':1})
    events=np.empty(8,dtype=[('is_4b','?'),('is_signal','?'),('weight','<f8')])
    events['is_4b']=[0,1,1,0,1,0,1,1];events['is_signal']=[0,0,1,0,1,0,0,1]
    events['weight']=[1,2,-.1,1,3,1,2,1]
    score=np.array([1,2,2,-1,3,0,.5,1.5],dtype=np.float32)
    curve=signal_efficiency(score,events)
    x,y=get_weights_by_sr_stats(SimpleNamespace(weights=events['weight'],is_4b=events['is_4b'],is_signal=events['is_signal']),score)
    np.testing.assert_array_equal(curve['signal_fraction'],np.interp(np.linspace(0,1,101),x,y))
    assert curve['negative_4b_events']==1
    weight=args.out/'weights';weight.write_bytes(b'opaque fixture weights; no inference')
    base=[];smooth=[]
    for seed in (0,1):
        model=store.put_model(weight,{'member':seed},{'hparams':{'step':1,'model':'FvTClassifier','model_seed':seed}},[split])
        sm=store.put_model(weight,{'member':seed},{'hparams':{'step':2,'model':'AttentionClassifier','model_seed':seed,'encoder_hash':'artifact:'+model}},[split])
        base.append((model,store.put_scores(model,split,score+seed*.1,'log_density_ratio')))
        smooth.append((sm,store.put_scores(sm,split,np.full(8,seed*.05,dtype=np.float32),'log_density_ratio')))
    event=store.put_array('event_metadata',{'dataset_id':dataset,'split_id':split},events)
    reader=SimpleNamespace(store=store,case=lambda key:{'completion':{'stage':1 if key=='base' else 2,'event_metadata_ids':{'X2':event}}},
        member_score_ids=lambda key,domain,**kw:base if key=='base' else smooth)
    before=store.storage_stats()
    for mode in ('max','mean'):
        value=upstream_efficiency(reader,'base','smooth',ensemble_mode=mode)
        arrays=np.array([score,score+np.float32(.1)-np.float32(.05)])
        expected=signal_efficiency(arrays.max(axis=0) if mode=='max' else arrays.mean(axis=0),events)
        np.testing.assert_array_equal(value['signal_fraction'],expected['signal_fraction'])
    assert store.storage_stats()==before
    import torch
    from ancillary_features import get_closest_dijet_masses
    torch.set_num_threads(1)
    generator=torch.Generator().manual_seed(21)
    jets=torch.rand((33,4,4),generator=generator)
    jets[:,0]=40+100*jets[:,0];jets[:,1]=2*jets[:,1]-1
    jets[:,2]=6*jets[:,2]-3;jets[:,3]=5+15*jets[:,3]
    features=jets.reshape(33,16);m0,m1=get_closest_dijet_masses(features)
    expected=-torch.sqrt((1-125/m0)**2+(1-125/m1)**2)
    np.testing.assert_array_equal(model_aware_score(features,batch_size=16),expected.numpy())
    figure='4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_max_vs_mean_sr_stats_eta=inf.pdf'
    case_ids=[f'base-{i}' for i in range(100)]
    nodes={key:{'id':key,'stage':1,'axes':{'mother_seed':i,'signal':'HH4b','epsilon':'0.01'}} for i,key in enumerate(case_ids)}
    full=SimpleNamespace(store=store,registry=SimpleNamespace(nodes=nodes,plan={'evaluation_bindings':[
        {'id':figure,'generator':'run_files/figure_scripts/signal_concentration.py','required_case_ids':case_ids}]}),
        case=lambda key:{'completion':{'stage':1,'event_metadata_ids':{'X2':event}}},
        member_score_ids=lambda *a,**kw:base)
    assert render(full,{'synthetic':True},args.out/'full-efficiency')==[figure]
    assert (args.out/'full-efficiency'/figure).stat().st_size>0
    assert store.storage_stats()==before
    calibration=weighted_calibration(np.array([1.,2.,3.,4.]),np.array([10.,20.,30.,40.]),np.array([1.,2.,1.,2.]),nbins=2)
    np.testing.assert_allclose(calibration['base_mean'],[5/3,11/3])
    np.testing.assert_allclose(calibration['ratio_mean'],[50/3,110/3])
    hist=class_histograms(score,events,np.linspace(-1,3,5),np.full(8,2.))
    np.testing.assert_array_equal(hist['reweighted3b'],2*hist['3b'])
    np.testing.assert_array_equal(hist['reweighted3b_variance'],4*hist['3b_variance'])
    overlay=RegionOverlay(store)
    assert overlay._record('region',{'fixture':True},{'log_tau_s':1.,'log_tau_c':None}) in overlay.records
    assert store.storage_stats()==before
    # Exercise all score-only illustration paths using explicit source-bound arrays.
    n=256;ds=store.put_dataset(sha(b'illustration-fixture'),n*2,{'fixture':True})
    splits={d:store.put_split(ds,d,np.arange(i*n,(i+1)*n,dtype=np.int64),{'algorithm':'fixture','seed':1,'version':1}) for i,d in enumerate(('X1','X2'))}
    ev=np.empty(n,dtype=[('is_4b','?'),('is_signal','?'),('weight','<f8')]);ev['is_4b']=np.arange(n)%2==1;ev['is_signal']=(np.arange(n)%8==1);ev['weight']=1.
    event_ids={d:store.put_array('event_metadata',{'dataset_id':ds,'split_id':split},ev) for d,split in splits.items()}
    models=[];scores={};sm_models={};sm_scores={}
    for seed in (0,1):
        m=store.put_model(weight,{'fixture_member':seed},{'hparams':{'step':1,'model':'FvTClassifier','model_seed':seed}},[splits['X1']]);models.append(m)
        for d,split in splits.items():scores[(seed,d)]=store.put_scores(m,split,np.linspace(-1,2,n,dtype=np.float32)+seed*.1,'log_density_ratio')
    for eta in ('0.1','0.5','2.0'):
        for seed,m in enumerate(models):
            sm=store.put_model(weight,{'fixture_smear':eta,'member':seed},{'hparams':{'step':2,'model':'AttentionClassifier','model_seed':seed,'encoder_hash':'artifact:'+m}},[splits['X1']]);sm_models[(eta,seed)]=sm
            for d,split in splits.items():sm_scores[(eta,seed,d)]=store.put_scores(sm,split,(float(eta)*.05*np.sin(np.linspace(0,6,n))).astype(np.float32),'log_density_ratio')
    parent_groups={'inf':['base']}|{eta:['base','sm-'+eta] for eta in ('0.1','0.5','2.0')}
    nodes={'base':{'stage':1,'requires':[],'axes':{}}}|{'sm-'+eta:{'stage':2,'requires':['base'],'axes':{'eta':eta}} for eta in ('0.1','0.5','2.0')}
    case_values={'base':{'completion':{'stage':1,'event_metadata_ids':event_ids}}}|{'sm-'+eta:{'completion':{'stage':2,'event_metadata_ids':event_ids}} for eta in ('0.1','0.5','2.0')}
    diagnostic={'classifier':{'upstream_nodes':['base','sm-2.0'],'member_seeds':[0,1]},'null_overlap':{'upstream_nodes_by_eta':parent_groups},
        'base_CR_histogram':{'training_nodes':[]},'smearing_tail':{'upstream_nodes_by_eta':{eta:parent_groups[eta] for eta in ('0.1','0.5','2.0')}},'original_vs_representation':{'training_nodes':[]}}
    def member_pairs(key,domain,member_seeds=None):
        selected=[0,1] if member_seeds is None else member_seeds
        if key=='base':return [(models[s],scores[(s,domain)]) for s in selected]
        eta=key[3:];return [(sm_models[(eta,s)],sm_scores[(eta,s,domain)]) for s in selected]
    mock=SimpleNamespace(store=store,registry=SimpleNamespace(nodes=nodes,plan={'region_recipe':{'version':'sr_quantile_cr_complement_v2'},'diagnostic_bindings':diagnostic}),case=lambda k:case_values[k],member_score_ids=member_pairs)
    mock.aggregate=lambda key,domain,**kw:(np.linspace(-.2,.2,n,dtype=np.float32),{'synthetic_rule':kw['aggregation']})
    for eta,parents in parent_groups.items():
        key='cr-'+eta;view=upstream_view(mock,parents);nodes[key]={'stage':3,'requires':parents,'axes':{'eta':eta}}
        case_values[key]={'completion':{'stage':3,'model_ids':[models[0],models[1]]},'task':{'upstream':{'region_id':view['region']['id']}}};diagnostic['base_CR_histogram']['training_nodes'].append(key)
    for architecture in ('FvTClassifier','AttentionClassifier'):
        key='repr-'+architecture;parents=parent_groups['2.0'];view=upstream_view(mock,parents,sr_fraction=.05,member_seeds=[1])
        nodes[key]={'stage':3,'requires':parents,'axes':{'model':architecture}}
        case_values[key]={'completion':{'stage':3,'model_ids':[models[1]]},'task':{'upstream':{'region_id':view['region']['id']}}};diagnostic['original_vs_representation']['training_nodes'].append(key)
    unchanged=store.storage_stats();illustrations=args.out/'illustrations';illustrations.mkdir()
    classifier(mock,illustrations);overlap(mock,illustrations);calibration_plot(mock,illustrations,'mean_probability');tail(mock,illustrations);representation(mock,illustrations,'mean_probability')
    assert len(list(illustrations.glob('*.pdf')))==5 and store.storage_stats()==unchanged
    asset=args.out/'asset';asset.mkdir();(asset/'toy.pdf').write_bytes(b'fixture PDF bytes')
    expected={'static_artifacts':{'toy.pdf':{'sha256':sha(b'fixture PDF bytes')}}}
    copied=args.out/'copied';copied.mkdir();assert copy_static(expected,asset,copied)['toy.pdf']==sha(b'fixture PDF bytes')
    # A self-contained plotting fixture exercises both eta and coverage guards.
    d=args.out/'diagnostic';(d/'cases').mkdir(parents=True)
    ids=[]
    for eta in ('2.0','inf'):
        for mother in (0,1):
            key=sha(canonical([eta,mother]));ids.append(key)
            value={'case_id':key,'axes':{'eta':eta,'mother_seed':mother},
                   'rule_metrics':{'single':{'pull':(np.linspace(-1,1,8)+mother*.1).tolist()}}}
            (d/'cases'/(key+'.json')).write_text(json.dumps({'value':value,'checksum':sha(canonical(value))}))
    manifest={'case_ids':ids,'nbins':8};summary={'manifest_sha256':sha(canonical(manifest))}
    (d/'training-plan.json').write_text(json.dumps(manifest));(d/'summary.json').write_text(json.dumps(summary))
    (d/'completion.json').write_text(json.dumps({'case_count':4,'summary_sha256':sha(canonical(summary))}))
    command=[sys.executable,str(ROOT/'phase5/render_campaign_pulls.py'),'--diagnostics',str(d),'--rule','single']
    subprocess.run(command+['--scope','pilot','--output',str(args.out/'pulls')],check=True)
    assert (args.out/'pulls/pilot_pull_vs_sr_stats_SR_size_0.2.pdf').stat().st_size>0
    bad=subprocess.run(command+['--scope','draft','--output',str(args.out/'invalid-draft')],capture_output=True,text=True)
    assert bad.returncode!=0 and not (args.out/'invalid-draft').exists()
    print(json.dumps({'status':'PASS','signed_efficiency_matches_legacy':True,'paired_max_and_mean_exact':True,
        'analytic_mass_score_chunking_exact':True,'full_efficiency_declared_coverage':True,'caption_calibration_quantiles_weighted':True,'weighted_histogram_variances':True,'region_overlay_no_store_writes':True,'score_only_illustrations':True,'static_copy_checksum':True,'store_unchanged':True,'pull_plot_new_root':True,'incomplete_draft_coverage_rejected':True}),flush=True)

if __name__=='__main__':main()
