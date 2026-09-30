"""Derive source-bound figure data from retained member scores, without writes."""
import numpy as np
from .regions import derive_region_scores


def signal_efficiency(score,events,points=None):
    score=np.asarray(score);points=np.linspace(0,1,101) if points is None else np.asarray(points)
    if score.shape!=events.shape or not np.isfinite(score).all():raise ValueError('Misaligned or nonfinite scores')
    weights=events['weight'].astype(np.float64)
    if not np.isfinite(weights).all():raise ValueError('Nonfinite physical weights')
    order=np.argsort(score,kind='stable')[::-1]
    w4=weights[order]*events['is_4b'][order];ws=weights[order]*events['is_signal'][order]
    if w4.sum()<=0 or ws.sum()<=0:raise ValueError('Efficiency requires positive 4b and signal totals')
    x=np.cumsum(w4)/w4.sum();y=np.cumsum(ws)/ws.sum()
    return {'four_b_fraction':points.tolist(),'signal_fraction':np.interp(points,x,y).tolist(),
            'negative_4b_events':int(np.count_nonzero(w4<0)),
            'recipe':'stable descending score order, signed cumulative physical weights, legacy np.interp'}


def upstream_efficiency(reader,base_case,smeared_case=None,*,ensemble_mode='max'):
    base=reader.case(base_case)
    if base['completion']['stage']!=1:raise ValueError('Expected a base ensemble')
    pairs=reader.member_score_ids(base_case,'X2')
    seeds=[reader.store.read(model,'model')['identity']['training_recipe']['hparams']['model_seed'] for model,_ in pairs]
    smooth=None
    if smeared_case is not None:
        if reader.case(smeared_case)['completion']['stage']!=2:raise ValueError('Expected a smeared ensemble')
        smooth=[key for _,key in reader.member_score_ids(smeared_case,'X2',member_seeds=seeds)]
    score,definition=derive_region_scores(reader.store,[key for _,key in pairs],smooth,ensemble_mode=ensemble_mode)
    event_id=base['completion']['event_metadata_ids']['X2']
    if reader.store.read(event_id)['identity']['split_id']!=definition['split_id']:raise ValueError('Event order differs')
    result=signal_efficiency(score,reader.store.load_array(event_id,'event_metadata'))
    return {**result,'base_case_id':base_case,'smeared_case_id':smeared_case,'member_seeds':seeds,
            'ensemble_mode':ensemble_mode,'definition':definition,'event_metadata_id':event_id}


def model_aware_score(features,batch_size=32768):
    import torch
    from ancillary_features import get_closest_dijet_masses
    features=torch.as_tensor(features,dtype=torch.float32)
    score=np.empty(len(features),dtype=np.float32)
    with torch.inference_mode():
        for begin in range(0,len(features),batch_size):
            m0,m1=get_closest_dijet_masses(features[begin:begin+batch_size])
            value=-torch.sqrt((1-125/m0)**2+(1-125/m1)**2)
            score[begin:begin+batch_size]=value.cpu().numpy()
    if not np.isfinite(score).all():raise ValueError('Nonfinite model-aware mass score')
    return score


def model_aware_efficiency(reader,base_case):
    from .bound_tasks import resolve_source_pointer
    from constants import FEATURES
    value=reader.case(base_case);source=resolve_source_pointer(reader.store,value['task']['source'])
    rows=source.indices('X2');frame=source.full_source[rows].fetch_data()
    if len(frame)!=len(rows):raise ValueError('Model-aware feature coverage differs')
    score=model_aware_score(frame[FEATURES].values)
    event_id=value['completion']['event_metadata_ids']['X2']
    return {**signal_efficiency(score,reader.store.load_array(event_id,'event_metadata')),
            'base_case_id':base_case,'event_metadata_id':event_id,
            'score_recipe':'negative sqrt((1-125/m0)^2+(1-125/m1)^2), closest dijet-mass pair'}
