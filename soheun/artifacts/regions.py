"""Derive region scores in memory and freeze X1-defined thresholds/provenance."""
import hashlib
import inspect
from types import SimpleNamespace
import numpy as np


class _WeightedEvents(SimpleNamespace):
    def __len__(self):
        return len(self.weights)


def derive_region_scores(store, base_score_ids, smeared_score_ids=None, *, ensemble_mode='max', use_logits=True):
    if not base_score_ids or len(set(base_score_ids))!=len(base_score_ids):
        raise ValueError('Nonempty distinct base members are required')
    if smeared_score_ids is not None and len(smeared_score_ids)!=len(base_score_ids):
        raise ValueError('One paired smeared score per base member is required')
    if ensemble_mode not in ('max','mean'):
        raise ValueError('Unsupported region aggregation')
    records=[store.read(key,'scores') for key in base_score_ids]
    split_id=records[0]['identity']['split_id']
    split=store.read(split_id,'split')
    dataset_id=split['identity']['dataset_id']
    values=[];base_models=[];smeared_models=[];recipes=[]
    for i,record in enumerate(records):
        ident=record['identity']
        if ident['split_id']!=split_id or ident['representation']!='log_density_ratio':
            raise ValueError('Base scores need common ordered events and raw log ratios')
        model=store.read(ident['owner_id'],'model')
        hp=model['identity']['training_recipe'].get('hparams',{})
        if hp.get('model')!='FvTClassifier' or int(hp.get('step',0))!=1:
            raise ValueError('Region base scores require Step-1 FvT models')
        base_models.append(model['id'])
        member=store.load_array(base_score_ids[i],'scores')
        if member.dtype!=np.float32 or member.shape!=(split['payload']['shape'][0],):
            raise ValueError('Base score array violates the evaluation layout')
        profile={'base':ident.get('inference_recipe')}
        if smeared_score_ids is not None:
            other=store.read(smeared_score_ids[i],'scores')['identity']
            if other['split_id']!=split_id or other['representation']!='log_density_ratio':
                raise ValueError('Smeared scores need matching ordered events')
            sm=store.read(other['owner_id'],'model')
            shp=sm['identity']['training_recipe'].get('hparams',{})
            if shp.get('model')!='AttentionClassifier' or int(shp.get('step',0))!=2 or shp.get('encoder_hash')!='artifact:'+model['id']:
                raise ValueError('Smeared member is not paired with this encoder')
            smeared_models.append(sm['id'])
            smoothed=store.load_array(smeared_score_ids[i],'scores')
            if smoothed.dtype!=np.float32 or smoothed.shape!=member.shape:
                raise ValueError('Smeared score array violates the evaluation layout')
            member=member-smoothed
            profile['smeared']=other.get('inference_recipe')
        if not np.isfinite(member).all():raise ValueError('Nonfinite region score')
        if not use_logits:
            # Preserve the existing compute_sr_stats probability-space option.
            with np.errstate(over='ignore'):
                member=1/(1+np.exp(-member))
        values.append(member);recipes.append(profile)
    if len(set(base_models))!=len(base_models):
        raise ValueError('The same base model cannot count as multiple ensemble members')
    result=np.max(values,axis=0) if ensemble_mode=='max' else np.mean(values,axis=0)
    definition={'dataset_id':dataset_id,'split_id':split_id,'base_models':base_models,
                'smeared_models':smeared_models,'ensemble_mode':ensemble_mode,
                'score_space':'log_psi' if use_logits else 'sigmoid_log_psi',
                'inference_recipes':recipes}
    return result,definition


def define_regions(store, base_X1_scores, event_metadata_id, smeared_X1_scores=None, *, sr_fraction, cr_fraction, ensemble_mode='max', quantile_recipe='sr_quantile_cr_complement_v2'):
    from signal_region import get_SR_CR_cut
    if quantile_recipe not in ('existing_get_SR_CR_cut_v1','sr_quantile_cr_complement_v2'):
        raise ValueError('Unsupported region quantile recipe')
    if not 0<sr_fraction<1 or not 0<cr_fraction<=1-sr_fraction+1e-12:
        raise ValueError('Invalid requested SR/CR fractions')
    values,definition=derive_region_scores(store,base_X1_scores,smeared_X1_scores,ensemble_mode=ensemble_mode)
    split=store.read(definition['split_id'],'split')
    if split['identity']['name']!='X1':raise ValueError('Regions must be defined on X1')
    metadata=store.read(event_metadata_id,'event_metadata')
    if metadata['identity'].get('split_id')!=definition['split_id'] or metadata['identity'].get('dataset_id')!=definition['dataset_id']:
        raise ValueError('Event metadata is not aligned with X1 scores')
    events=store.load_array(event_metadata_id,'event_metadata')
    if events.shape!=values.shape or not {'is_4b','weight'}<=set(events.dtype.names or ()):
        raise ValueError('Missing aligned physical weights/class labels')
    labels=events['is_4b'];weights=events['weight']
    if not np.isin(labels,[0,1]).all() or not np.isfinite(weights).all():raise ValueError('Invalid event metadata')
    labels=labels.astype(bool)
    total=float(weights[labels].sum())
    if not np.isfinite(total) or total<=0:raise ValueError('X1 signed 4b total must be positive')
    cuts=get_SR_CR_cut(values,_WeightedEvents(weights=weights,is_4b=labels),
                       {'4b_in_SR':sr_fraction,'4b_in_CR':cr_fraction})
    if not np.isfinite(cuts).all():raise ValueError('Nonfinite region thresholds')
    complement=(quantile_recipe=='sr_quantile_cr_complement_v2'
                and abs(cr_fraction-(1-sr_fraction))<=1e-12)
    identity={'definition':definition,'base_score_ids':list(base_X1_scores),
              'smeared_score_ids':list(smeared_X1_scores or []),'event_metadata_id':event_metadata_id,
              'requested_sr_fraction':sr_fraction,'requested_cr_fraction':cr_fraction,
              'quantile_recipe':quantile_recipe,
              'quantile_code_sha256':hashlib.sha256(inspect.getsource(get_SR_CR_cut).encode()).hexdigest(),
              'quantile_fraction_clip':[.001,.999]}
    if quantile_recipe=='sr_quantile_cr_complement_v2':
        identity['cr_lower_bound']='none' if complement else 'finite'
    return store._record('region',identity,{'log_tau_s':float(cuts[0]),'log_tau_c':None if complement else float(cuts[1]),
                                          'X1_4b_weight_total':total,'X1_rows':len(values)})


def classify_X2(store, region_id, base_X2_scores, smeared_X2_scores=None):
    region=store.read(region_id,'region')
    expected=region['identity']['definition']
    score,actual=derive_region_scores(store,base_X2_scores,smeared_X2_scores,
                                     ensemble_mode=expected['ensemble_mode'])
    if store.read(actual['split_id'],'split')['identity']['name']!='X2':
        raise ValueError('Expected X2 evaluation scores')
    if any(actual[k]!=expected[k] for k in ('dataset_id','base_models','smeared_models','ensemble_mode','score_space','inference_recipes')):
        raise ValueError('X2 scores do not match the frozen region definition')
    lower,upper=region['payload']['log_tau_c'],region['payload']['log_tau_s']
    sr=score>=upper
    cr=~sr if lower is None else (score>=lower)&~sr
    return {'log_psi':score,'SR':sr,'CR':cr,'neither':~(sr|cr),'region_id':region_id}
