"""Pre-power null diagnostics; transient X1 CR scores preserve legacy train binning."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from constants import FEATURES
from utils import get_quantiles_with_weights
from .bound_tasks import resolve_source_pointer
from .regions import derive_region_scores
from .model_loading import load_model
from .training_store import aggregate_log_ratios
from .runtime_policy import validated_gpu_runtime
from .export_stage import _profile


def count_and_shape(train_score,train_weight,train_log_ratio,score,labels,weights,log_ratio,*,nbins=64,signal_truth=None,background_only=False):
    """Legacy X1 reweighted-3b quantiles and sqrt(sum(w4²)+sum(q3²)) pulls."""
    if nbins<1:raise ValueError('Positive bin count required')
    labels=np.asarray(labels,dtype=bool);weights=np.asarray(weights,dtype=np.float64)
    if background_only:
        if signal_truth is None:raise ValueError('Background-only diagnostics require signal truth')
        truth=np.asarray(signal_truth,dtype=bool)
        if truth.shape!=labels.shape:raise ValueError('Signal truth is not aligned')
        target4=labels&~truth
    else:target4=labels
    w3=weights[~labels];w4=weights[target4];all4=weights[labels]
    if np.any(w3<0) or w4.sum()<=0:raise ValueError('Invalid physical-weight normalization')
    logq=np.full(w3.shape,-np.inf);positive=w3>0
    logq[positive]=np.asarray(log_ratio,dtype=np.float64)[~labels][positive]+np.log(w3[positive])
    if not positive.any():raise ValueError('No positive 3b weight')
    shift=max(float(logq.max()),float(np.log(np.abs(all4[all4!=0])).max()))
    q=np.exp(logq-shift)
    scaled4=np.sign(w4)*np.exp(np.where(w4!=0,np.log(np.maximum(np.abs(w4),np.finfo(float).tiny)),-np.inf)-shift)
    total4=scaled4.sum()
    if not total4>0:raise ValueError('4b normalizer underflows under extreme ratio tails')
    error=float(q.sum()/total4-1)
    if not np.isfinite(error):raise ValueError('Nonfinite relative count error')
    train_weight=np.asarray(train_weight,dtype=np.float64);active=train_weight>0
    if np.any(train_weight<0) or not active.any():raise ValueError('Invalid X1 3b binning weights')
    terms=np.asarray(train_log_ratio,dtype=np.float64)[active]+np.log(train_weight[active])
    bin_weights=np.exp(terms-terms.max())
    bins=get_quantiles_with_weights(np.asarray(train_score)[active],bin_weights,np.linspace(0,1,nbins+1))
    h3=np.histogram(np.asarray(score)[~labels],bins=bins,weights=q)[0]
    h4=np.histogram(np.asarray(score)[target4],bins=bins,weights=scaled4)[0]
    # Legacy pull_bg4b changes the numerator while retaining total-4b variance.
    scaled_all4=np.sign(all4)*np.exp(np.where(all4!=0,np.log(np.maximum(np.abs(all4),np.finfo(float).tiny)),-np.inf)-shift)
    variance=np.histogram(np.asarray(score)[~labels],bins=bins,weights=q*q)[0]+np.histogram(np.asarray(score)[labels],bins=bins,weights=scaled_all4*scaled_all4)[0]
    valid=variance>0;pull=np.zeros(nbins);pull[valid]=(h4[valid]-h3[valid])/np.sqrt(variance[valid])
    return {'relative_count_error':error,'shape_error':float(np.sqrt(np.mean(pull*pull))) if valid.all() else None,
            'pull':[float(p) if ok else None for p,ok in zip(pull,valid)],'bin_edges':bins.tolist(),
            'zero_variance_bins':int((~valid).sum()),'nbins':nbins,
            'X2_SR_outside_X1_bin_range':int(np.count_nonzero((score<bins[0])|(score>bins[-1]))),
            'physical_4b_total':float(all4.sum()),'target_4b_total':float(w4.sum()),
            'target':'background_4b' if background_only else 'all_4b',
            'common_log_weight_scale':shift,
            'binning':'X1 SR reweighted 3b quantiles; finite edges as in the legacy train-binning figure',
            'variance':'sum 4b physical weight squared + sum reweighted 3b weight squared'}


@validated_gpu_runtime
def diagnose_case(reader,case_id,*,device='cpu',batch_size=32768,nbins=64,analysis='ensemble_selection'):
    node=reader.registry.nodes[case_id];a=node['axes']
    if (a.get('signal')!='HH4b' or float(a['eta']) not in (2.,float('inf'))
            or float(a['epsilon']) not in (0.,.005,.0075,.01,.02)
            or float(a['sr_fraction']) not in (.05,.1,.15,.2)):
        raise ValueError('Diagnostics are restricted to the approved HH4b comparison cells')
    if analysis not in ('ensemble_selection','extrapolation_bias'):raise ValueError('Unknown diagnostic analysis')
    if analysis=='ensemble_selection' and (float(a['epsilon'])!=0 or float(a['sr_fraction'])!=.2):
        raise ValueError('Ensemble selection requires null SR=0.2 cells')
    value=reader.case(case_id);store=reader.store;upstream=value['task']['upstream']
    region=store.read(upstream['region_id'],'region')
    x1_score,_=derive_region_scores(store,region['identity']['base_score_ids'],region['identity']['smeared_score_ids'] or None,
                                    ensemble_mode=region['identity']['definition']['ensemble_mode'])
    event1=store.load_array(region['identity']['event_metadata_id'],'event_metadata')
    keep1=(x1_score>=region['payload']['log_tau_s'])&~event1['is_4b']
    source=resolve_source_pointer(store,value['task']['source'])
    rows=source.indices('X1')[keep1]
    frame=source.full_source[rows].fetch_data()
    if len(frame)!=len(rows):raise ValueError('X1 SR feature ordering is incomplete')
    x=torch.tensor(frame[FEATURES].values,dtype=torch.float32).contiguous()
    if not torch.isfinite(x).all():raise ValueError('Nonfinite X1 features')
    pairs=reader.member_score_ids(case_id,'X2');seeds=[];pred1=[];pred2=[];profiles=[]
    masks=reader.held_out_regions(case_id);keep2=masks['SR']
    event2=store.load_array(value['completion']['event_metadata_ids']['X2'],'event_metadata')[keep2]
    for model_id,score_id in pairs:
        hp=store.read(model_id,'model')['identity']['training_recipe']['hparams']
        if hp['model']!='FvTClassifier':raise ValueError('Null development cases require the declared raw FvT architecture')
        seeds.append(hp['model_seed']);profiles.append(_profile(hp,device,batch_size))
        model=load_model(store,model_id,device=device);scores=np.empty(len(x),dtype=np.float32)
        with torch.inference_mode():
            for begin in range(0,len(x),batch_size):
                logits=model(x[begin:begin+batch_size].to(device))
                scores[begin:begin+batch_size]=(logits[:,1]-logits[:,0]).cpu().numpy()
        if not np.isfinite(scores).all():raise ValueError('Nonfinite transient X1 SR 3b scores')
        pred1.append(scores);pred2.append(store.load_array(score_id,'scores')[keep2]);del model
    if 0 not in seeds:raise ValueError('Fixed member 0 is missing')
    def metric(g1,g2):
        return count_and_shape(x1_score[keep1],event1['weight'][keep1],g1,masks['log_psi'][keep2],
                               event2['is_4b'],event2['weight'],g2,nbins=nbins,
                               signal_truth=event2['is_signal'] if analysis=='extrapolation_bias' else None,
                               background_only=analysis=='extrapolation_bias' and float(a['epsilon'])>0)
    members={str(seed):metric(p1,p2) for seed,p1,p2 in zip(seeds,pred1,pred2)}
    results={'single':members['0']}
    for rule in ('mean_probability','mean_log_density_ratio','mean_density_ratio'):
        results[rule]=metric(aggregate_log_ratios(pred1,rule),aggregate_log_ratios(pred2,rule))
    return {'case_id':case_id,'axes':a,'analysis':analysis,'stage_completion_id':value['completion_id'],'member_seeds':seeds,
            'member_metrics':members,'rule_metrics':results,'region_id':upstream['region_id'],
            'X1_SR_3b_scores':'computed transiently from frozen best models; not added to the permanent store',
            'X1_inference_profiles':profiles,'X1_prediction_sha256':[hashlib.sha256(v.tobytes()).hexdigest() for v in pred1],
            'X2_score_ids':[s for _,s in pairs]}


def summarize_diagnostics(cases):
    groups={}
    for case in cases:groups.setdefault(case['axes']['eta'],[]).append(case)
    rows=[];correlations={}
    for eta,group in groups.items():
        seeds=group[0]['member_seeds'];k=len(seeds)
        if any(c['member_seeds']!=seeds for c in group):raise ValueError('Member sets differ across diagnostic cases')
        if len({c['axes']['mother_seed'] for c in group})!=len(group):raise ValueError('Duplicate diagnostic mother seed')
        matrix=np.array([[c['member_metrics'][str(s)]['relative_count_error'] for s in seeds] for c in group])
        sigma=float(np.var(matrix,axis=0).mean());centered=matrix-matrix.mean(axis=0)
        cov=centered.T@centered/len(group)
        rho=float((cov.sum()-np.trace(cov))/(k*(k-1))/sigma) if len(group)>1 and k>1 and sigma>0 else None
        correlations[eta]={'empirical_member_error_correlation':rho,'exchangeable_member_variance':sigma,
            'variance_baseline':'Var1 pooled across all members; single row uses fixed member 0',
            'mother_seeds':sorted(c['axes']['mother_seed'] for c in group)}
        for rule in group[0]['rule_metrics']:
            errors=np.array([c['rule_metrics'][rule]['relative_count_error'] for c in group])
            shapes=[c['rule_metrics'][rule]['shape_error'] for c in group]
            mean=float(errors.mean());var=float(errors.var());rms=float(np.sqrt(np.mean(errors*errors)))
            effective=(k*var/sigma-1)/(k-1) if len(group)>1 and k>1 and sigma>0 and rule!='single' else None
            projected_var=None if effective is None else sigma*(effective+(1-effective)/15)
            predicted=None if projected_var is None or projected_var<0 else float(np.sqrt(mean*mean+projected_var))
            rows.append({'eta':eta,'rule':rule,'n':len(group),'K':1 if rule=='single' else k,
                'mean_count_error':mean,'rms_count_error':rms,
                'mean_shape_error':float(np.mean(shapes)) if all(x is not None for x in shapes) else None,
                'invalid_shape_cases':sum(x is None for x in shapes),'effective_rho_from_variance_ratio':effective,
                'predicted_K15_RMS':predicted,'predicted_fractional_improvement':None if predicted is None or rms==0 else 1-predicted/rms,
                'projection_assumption':'unchanged bias and exchangeable linear averaging; approximate for mean-probability/geometric-mean rules'})
    return {'rows':rows,'correlations':correlations,'decision':'User decision; advisory thresholds are not automatic acceptance',
            'limitation':'K=15 prediction is a variance-model diagnostic, not a measured 15-member result'}


def decision_guidance(summary):
    rows=summary['rows'];etas={r['eta'] for r in rows}
    if {float(e) for e in etas}!={2.,float('inf')}:
        return {'status':'INCOMPLETE_ETA_COVERAGE','decision':'User decision pending'}
    coverage=summary.get('correlations',{})
    if (any(r['n']!=100 for r in rows)
            or any(coverage.get(e,{}).get('mother_seeds')!=list(range(100)) for e in etas)):
        return {'status':'INSUFFICIENT_N','required_mother_seeds':list(range(100)),
                'decision':'User decision pending; tier-A diagnostics have no indicative rule'}
    rules=('mean_probability','mean_log_density_ratio','mean_density_ratio')
    by_rule={rule:[r for r in rows if r['rule']==rule] for rule in rules}
    if any(len(group)!=2 or any(r['mean_shape_error'] is None for r in group) for group in by_rule.values()):
        return {'status':'INCOMPLETE_SHAPE_METRICS','decision':'User decision pending'}
    counts={rule:float(np.mean([r['rms_count_error'] for r in group])) for rule,group in by_rule.items()}
    shapes={rule:float(np.mean([r['mean_shape_error'] for r in group])) for rule,group in by_rule.items()}
    count_rule=min(rules,key=counts.get);shape_rule=min(rules,key=shapes.get)
    candidate=shape_rule if shapes[count_rule]>1.05*shapes[shape_rule] else count_rule
    predictions=[r['predicted_K15_RMS'] for r in by_rule[candidate]]
    gain=None if any(v is None for v in predictions) or counts[candidate]==0 else 1-float(np.mean(predictions))/counts[candidate]
    return {'status':'ADVISORY_ONLY','average_RMS_by_rule':counts,'average_shape_by_rule':shapes,
            'indicative_rule':candidate,'count_RMS_ties':[r for r in rules if counts[r]==counts[count_rule]],
            'predicted_average_RMS_reduction_K5_to_K15':gain,
            'indicative_extension_to_15':None if gain is None else gain>=.10,
            'final_decision':'USER_REQUIRED','thresholds_binding':False,
            'projection_scope':'Average RMS across the two eta values; also inspect each eta separately'}
