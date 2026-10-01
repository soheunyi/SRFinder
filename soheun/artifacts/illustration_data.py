"""Read-only data for the manuscript's fixed methodological illustrations."""
import numpy as np
from .training_store import canonical,sha,aggregate_log_ratios
from .regions import define_regions,classify_X2,derive_region_scores


class RegionOverlay:
    """Use the production region recipe without persisting figure-only records."""
    def __init__(self,store):self.store=store;self.records={}
    def read(self,key,kind=None):
        if key in self.records:return self.records[key]
        return self.store.read(key,kind)
    def load_array(self,*args):return self.store.load_array(*args)
    def _record(self,kind,identity,payload):
        if kind!='region':raise ValueError('Figure overlay accepts only region descriptors')
        value={'schema':1,'kind':kind,'identity':identity};key=sha(canonical(value))
        self.records[key]={**value,'id':key,'payload':payload};return key


def finite_ratio(log_ratio):
    with np.errstate(over='ignore'):value=np.exp(np.asarray(log_ratio,dtype=np.float64))
    if not np.isfinite(value).all():raise ValueError('Ratio tail exceeds plotting precision; inspect member scores')
    return value


def upstream_view(reader,case_ids,*,sr_fraction=.2,member_seeds=None):
    cases=[(key,reader.case(key)) for key in case_ids]
    base=[key for key,v in cases if v['completion']['stage']==1]
    smooth=[key for key,v in cases if v['completion']['stage']==2]
    if len(base)!=1 or len(smooth)>1:raise ValueError('One paired upstream source is required')
    pairs={domain:reader.member_score_ids(base[0],domain,member_seeds=member_seeds) for domain in ('X1','X2')}
    seeds=[reader.store.read(m,'model')['identity']['training_recipe']['hparams']['model_seed'] for m,_ in pairs['X1']]
    smooth_ids={domain:[s for _,s in reader.member_score_ids(smooth[0],domain,member_seeds=seeds)] if smooth else None for domain in ('X1','X2')}
    value=reader.case(base[0]);events={d:value['completion']['event_metadata_ids'][d] for d in ('X1','X2')}
    overlay=RegionOverlay(reader.store)
    recipe=reader.registry.plan.get('region_recipe')
    if not recipe or not recipe.get('version'):
        # Same rule as materialize_task: no silent fallback to the legacy recipe.
        raise ValueError('Illustrations require the plan to pin region_recipe')
    key=define_regions(overlay,[s for _,s in pairs['X1']],events['X1'],smooth_ids['X1'],
        sr_fraction=sr_fraction,cr_fraction=1-sr_fraction,quantile_recipe=recipe['version'])
    record=overlay.read(key);masks=classify_X2(overlay,key,[s for _,s in pairs['X2']],smooth_ids['X2'])
    x1,_=derive_region_scores(reader.store,[s for _,s in pairs['X1']],smooth_ids['X1'])
    scores=[reader.store.load_array(s,'scores') for _,s in pairs['X2']]
    return {'masks':masks,'X1_log_psi':x1,'events':reader.store.load_array(events['X2'],'event_metadata'),
        'base_first_log_ratio':scores[0],'base_mean_ratio':finite_ratio(aggregate_log_ratios(scores,'mean_density_ratio')),
        'region':record,'base_case_id':base[0],'base_models':[m for m,_ in pairs['X2']],
        'event_metadata_id':events['X2'],'member_seeds':seeds}


def weighted_calibration(base,ratio,weights,nbins=50):
    base=np.asarray(base,dtype=float);ratio=np.asarray(ratio,dtype=float);weights=np.asarray(weights,dtype=float)
    if not (base.shape==ratio.shape==weights.shape) or not np.isfinite([base,ratio,weights]).all():raise ValueError('Invalid calibration vectors')
    edges=np.quantile(base,np.linspace(0,1,nbins+1));index=np.clip(np.searchsorted(edges,base,side='right')-1,0,nbins-1)
    total=np.bincount(index,weights=weights,minlength=nbins)
    count=np.bincount(index,minlength=nbins);valid=total>0
    x=np.bincount(index,weights=weights*base,minlength=nbins)
    y=np.bincount(index,weights=weights*ratio,minlength=nbins)
    return {'base_mean':[float(a/b) if ok else None for a,b,ok in zip(x,total,valid)],
        'ratio_mean':[float(a/b) if ok else None for a,b,ok in zip(y,total,valid)],
        'event_counts':count.tolist(),'edges':edges.tolist(),
        'recipe':'50 equal-count quantile bins, physical-event-weighted means as stated in the draft caption'}


def class_histograms(values,events,bins,reweight=None):
    values=np.asarray(values);weight=events['weight'].astype(float);labels=events['is_4b'];truth=events['is_signal']
    groups={'3b':~labels,'background4b':labels&~truth,'signal':truth}
    result={}
    for key,mask in groups.items():
        w=weight[mask];result[key]=np.histogram(values[mask],bins=bins,weights=w)[0]
        result[key+'_variance']=np.histogram(values[mask],bins=bins,weights=w*w)[0]
    if reweight is not None:
        w=weight[~labels]*np.asarray(reweight)[~labels]
        result['reweighted3b']=np.histogram(values[~labels],bins=bins,weights=w)[0]
        result['reweighted3b_variance']=np.histogram(values[~labels],bins=bins,weights=w*w)[0]
    return result
