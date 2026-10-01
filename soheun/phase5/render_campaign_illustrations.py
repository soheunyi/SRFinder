"""Render declared methodological illustrations from the new store."""
import argparse,inspect,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from artifacts.evaluation import open_reader
from artifacts.illustration_data import upstream_view,finite_ratio,weighted_calibration,class_histograms
from artifacts.training_store import canonical,sha
from artifacts.bound_tasks import resolve_source_pointer
from artifacts.model_loading import load_model
from artifacts.runtime_policy import runtime_policy
from constants import FEATURES


NAMES={'classifier':'classifier_example_hist_exp=smeared_fvt_training_ensemble_signal=0.01.pdf',
       'overlap':'smearing_overlap_signal_ratio_0.0_seed_5.pdf','calibration':'base_and_CR_fvt_scores_hist_seed_5.pdf',
       'tail':'smearing_and_tails_signal_ratio_0.01_seed_50.pdf','representation':'on_which_to_learn.pdf','tsne':'tsne_original_repr.pdf'}


def save(fig,root,name,provenance):
    fig.tight_layout();fig.savefig(root/name);plt.close(fig)
    (root/(name+'.json')).write_text(json.dumps(provenance,indent=2)+'\n')


def reference(view):
    return {'region_id':view['region']['id'],'region_identity':view['region']['identity'],
            'region_thresholds':view['region']['payload'],'base_models':view['base_models'],
            'event_metadata_id':view['event_metadata_id'],'member_seeds':view['member_seeds']}


def classifier(reader,root):
    scope=reader.registry.plan['diagnostic_bindings']['classifier'];ids=scope['upstream_nodes']
    base=next(key for key in ids if reader.registry.nodes[key]['stage']==1)
    value=reader.case(base);events=reader.store.load_array(value['completion']['event_metadata_ids']['X2'],'event_metadata')
    pairs=reader.member_score_ids(base,'X2',member_seeds=scope['member_seeds']);fig,axes=plt.subplots(1,len(pairs),figsize=(15,5),squeeze=False)
    for ax,(model,score),seed in zip(axes[0],pairs,scope['member_seeds']):
        h=class_histograms(expit(reader.store.load_array(score,'scores')),events,np.linspace(0,1,100))
        for kind in ('3b','background4b','signal'):ax.stairs(h[kind],np.linspace(0,1,100),label=kind)
        ax.set(title=f'Member seed {seed}',xlabel='Classifier probability',ylabel='Weighted counts',yscale='log');ax.legend()
    save(fig,root,NAMES['classifier'],{'case_id':base,'member_seeds':scope['member_seeds'],'score_ids':[s for _,s in pairs],
        'note':'Fixed seeds 5/13; no old success/failure conclusion is imposed'})


def overlap(reader,root):
    groups=reader.registry.plan['diagnostic_bindings']['null_overlap']['upstream_nodes_by_eta']
    fig,axes=plt.subplots(2,3,figsize=(15,8));refs=[]
    for i,eta in enumerate(('inf','2.0','0.1')):
        view=upstream_view(reader,groups[eta]);events=view['events'];y=finite_ratio(view['masks']['log_psi'])
        # Stand-in for the true ratio: mean density ratio over all base members (user decision), not one member.
        x=view['base_mean_ratio']
        xb=np.linspace(*np.quantile(x,[.005,.999]),30);yb=np.linspace(*np.quantile(y,[.005,.999]),30);labels=events['is_4b']
        axes[0,i].hist2d(x[labels],y[labels],bins=(xb,yb),weights=events['weight'][labels]);axes[0,i].axhline(np.exp(view['region']['payload']['log_tau_s']),color='red',linestyle='--')
        for region in ('SR','CR'):
            mask=view['masks'][region]&labels;axes[1,i].hist(x[mask],bins=xb,weights=events['weight'][mask],histtype='step',label=region)
        axes[0,i].set(title=f'eta={eta}',ylabel='Learned SR score');axes[1,i].set(xlabel='Base density ratio (member mean)',ylabel='4b weighted counts');axes[1,i].legend()
        refs.append({'eta':eta,'view':reference(view),'base_axis':'mean_density_ratio over base members','base_axis_member_seeds':view['member_seeds']})
    save(fig,root,NAMES['overlap'],{'sources':refs,'region_rule':'Current pinned complement-CR recipe'})


def aggregate_cr(reader,key,rule):
    if rule!='single':return reader.aggregate(key,'X2',aggregation=rule)
    value=reader.case(key)
    seeds=[reader.store.read(m,'model')['identity']['training_recipe']['hparams']['model_seed'] for m in value['completion']['model_ids']]
    return reader.aggregate(key,'X2',aggregation='single',member_seeds=[0] if len(seeds)>1 else seeds)


def calibration(reader,root,rule):
    keys=reader.registry.plan['diagnostic_bindings']['base_CR_histogram']['training_nodes']
    cases={reader.registry.nodes[key]['axes']['eta']:key for key in keys}
    fig,axes=plt.subplots(2,3,figsize=(15,8));rows=[]
    for col,eta in enumerate(('inf','2.0','0.1')):
        key=cases[eta];value=reader.case(key);up=value['task']['upstream']
        parent=reader.registry.nodes[key]['requires'];view=upstream_view(reader,parent)
        if view['region']['id']!=up['region_id']:raise ValueError('Calibration region differs from trained CR')
        log_ratio,reduction=aggregate_cr(reader,key,rule);gamma=finite_ratio(log_ratio);events=view['events']
        # Same stand-in as the overlap panel: mean density ratio over all base members (user decision).
        base=view['base_mean_ratio']
        edges=np.linspace(base.min(),base.max(),80)
        for row,region in enumerate(('CR','SR')):
            mask=view['masks'][region];ax=axes[row,col]
            ax.hist2d(base[mask],gamma[mask],bins=(edges,edges));line=weighted_calibration(base[mask],gamma[mask],events['weight'][mask])
            valid=np.array([x is not None for x in line['base_mean']]);ax.plot(np.asarray(line['base_mean'],dtype=float)[valid],np.asarray(line['ratio_mean'],dtype=float)[valid],color='red')
            ax.plot([edges[0],edges[-1]],[edges[0],edges[-1]],'k--');ax.set(title=f'eta={eta}, {region}',xlabel='Base density ratio (member mean)',ylabel='CR estimate')
            rows.append({'case_id':key,'region':region,'calibration':line,'reduction':reduction,'view':reference(view),
                         'base_axis_member_seeds':view['member_seeds']})
    save(fig,root,NAMES['calibration'],{'rule':rule,'rows':rows,'x_axis':'mean_density_ratio over base members, same as the overlap panel (user decision; the older figure used one member)',
        'legacy_display_difference':'Caption-matching equal-count bins and physical-weighted means; older display used uniform bins/unweighted means'})


def tail(reader,root):
    groups=reader.registry.plan['diagnostic_bindings']['smearing_tail']['upstream_nodes_by_eta']
    fig,axes=plt.subplots(2,3,figsize=(15,8));refs=[]
    for i,eta in enumerate(('0.1','0.5','2.0')):
        view=upstream_view(reader,groups[eta]);values=finite_ratio(view['masks']['log_psi']);mask=view['masks']['SR'];events=view['events']
        bins=np.linspace(values[mask].min(),np.quantile(values[mask],.999),20)
        # Restrict the histogram itself to SR, retaining physical signed weights.
        h=class_histograms(values[mask],events[mask],bins)
        for kind in ('background4b','signal'):
            total=h[kind].sum();density=h[kind]/total/np.diff(bins) if total>0 else np.zeros(len(bins)-1)
            axes[0,i].stairs(density,bins,label=kind)
        denominator=h['background4b']+h['signal'];fraction=np.divide(h['signal'],denominator,out=np.full_like(denominator,np.nan),where=denominator!=0)
        axes[1,i].plot((bins[:-1]+bins[1:])/2,fraction);axes[0,i].set(title=f'eta={eta}',ylabel='Density');axes[0,i].legend();axes[1,i].set(xlabel='Learned SR score',ylabel='Signal fraction',ylim=(0,1))
        refs.append({'eta':eta,'view':reference(view),'bins':bins.tolist()})
    save(fig,root,NAMES['tail'],{'sources':refs,'CR_training_required':False})


def representation(reader,root,rule):
    keys=reader.registry.plan['diagnostic_bindings']['original_vs_representation']['training_nodes']
    views=[];fig,axes=plt.subplots(2,2,figsize=(16,7))
    for col,architecture in enumerate(('FvTClassifier','AttentionClassifier')):
        key=next(k for k in keys if reader.registry.nodes[k]['axes']['model']==architecture);value=reader.case(key);up=value['task']['upstream']
        view=upstream_view(reader,reader.registry.nodes[key]['requires'],sr_fraction=.05,member_seeds=[1])
        if view['region']['id']!=up['region_id']:raise ValueError('Representation comparison needs its single-member region')
        mask=view['masks']['SR'];x1=view['X1_log_psi'];cut=view['region']['payload']['log_tau_s'];bins=np.linspace(x1[x1>=cut].min(),x1[x1>=cut].max(),17)
        ratio,reduction=aggregate_cr(reader,key,rule);h=class_histograms(view['masks']['log_psi'][mask],view['events'][mask],bins,finite_ratio(ratio)[mask])
        for kind in ('3b','background4b','signal','reweighted3b'):axes[0,col].stairs(h[kind],bins,label=kind)
        bg=h['background4b'];q=h['reweighted3b'];valid=bg>0;centers=(bins[:-1]+bins[1:])/2
        r=np.divide(q,bg,out=np.full_like(q,np.nan),where=valid)
        var=np.divide(h['reweighted3b_variance'],bg*bg,out=np.zeros_like(q),where=valid)+np.divide(q*q*h['background4b_variance'],bg**4,out=np.zeros_like(q),where=valid)
        axes[1,col].errorbar(centers[valid],r[valid],yerr=np.sqrt(var[valid]),fmt='o',capsize=2);axes[1,col].axhline(1,color='gray',linestyle='--')
        axes[0,col].set(title='Original space' if col==0 else 'Representation space',ylabel='Weighted counts');axes[0,col].legend();axes[1,col].set(xlabel='Log learned SR score',ylabel='Estimated / true background')
        views.append({'case_id':key,'view':reference(view),'reduction':reduction,'bins':bins.tolist()})
    save(fig,root,NAMES['representation'],{'sources':views,'ratio_uncertainty':'Conditional delta-method variance using squared physical/reweighted event weights'})


def tsne(reader,root,device):
    from sklearn.manifold import TSNE
    import sklearn
    key=reader.registry.plan['diagnostic_bindings']['original_vs_representation']['training_nodes'][0]
    value=reader.case(key);view=upstream_view(reader,reader.registry.nodes[key]['requires'],sr_fraction=.05,member_seeds=[1]);positions=np.flatnonzero(view['masks']['CR'])
    if len(positions)<20000:raise ValueError('Declared t-SNE needs 20,000 CR events')
    chosen=np.sort(np.random.default_rng(42).choice(positions,20000,replace=False))
    source=resolve_source_pointer(reader.store,value['task']['source']);rows=source.indices('X2')[chosen];selected=source.full_source[rows]
    info=selected.to_dataset_info();events=view['events'][chosen]
    if not np.array_equal(info.file_idx,events['pool']) or not np.array_equal(info.inner_idx,events['pool_row']):raise ValueError('Sample feature and label ordering differ')
    frame=selected.fetch_data();features=torch.tensor(frame[FEATURES].values,dtype=torch.float32)
    model=load_model(reader.store,view['base_models'][0],device=device)
    with runtime_policy(device),torch.inference_mode():encoded=model.q_repr(features).numpy().reshape(20000,-1)
    parameters={'n_components':2,'perplexity':20,'learning_rate':'auto','init':'pca','early_exaggeration':12,'random_state':42,'metric':'euclidean'}
    parameters['max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter']=500
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for ax,array,title in zip(axes,(features.numpy(),encoded),('Original space','Representation space')):
        output=TSNE(**parameters).fit_transform(array)
        for label,mask in [('3b',~events['is_4b']),('background4b',events['is_4b']&~events['is_signal']),('signal',events['is_signal'])]:ax.scatter(output[mask,0],output[mask,1],s=20 if label=='signal' else 2,label=label,rasterized=True)
        ax.set_title(title);ax.legend()
    save(fig,root,NAMES['tsne'],{'case_id':key,'base_model':view['base_models'][0],'sampling_seed':42,'sample_order':'canonical source order',
        'sample_event_sha256':sha(np.column_stack((events['pool'],events['pool_row'])).tobytes()),'parameters':parameters,'sklearn_version':sklearn.__version__})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execution',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--only',nargs='+',choices=NAMES,default=list(NAMES));ap.add_argument('--rule',choices=['single','mean_probability','mean_log_density_ratio','mean_density_ratio'],required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=ap.parse_args();torch.set_num_threads(1)
    reader,manifest=open_reader(args.execution);declared={b['id'] for b in reader.registry.plan['evaluation_bindings']}
    if any(NAMES[k] not in declared for k in args.only):raise ValueError('Illustration is outside the frozen draft')
    args.output.mkdir(parents=True,exist_ok=False)
    for kind in args.only:
        if kind=='classifier':classifier(reader,args.output)
        elif kind=='overlap':overlap(reader,args.output)
        elif kind=='calibration':calibration(reader,args.output,args.rule)
        elif kind=='tail':tail(reader,args.output)
        elif kind=='representation':representation(reader,args.output,args.rule)
        else:tsne(reader,args.output,args.device)
    (args.output/'provenance.json').write_text(json.dumps({'training_manifest_sha256':sha(canonical(manifest)),
        'rule':args.rule,'figures':[NAMES[k] for k in args.only]},indent=2)+'\n')

if __name__=='__main__':main()
