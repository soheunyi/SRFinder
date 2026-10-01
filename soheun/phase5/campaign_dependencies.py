"""Logical dependencies for the reviewed draft grid; never submits training.

Case/member IDs here are planning identities, not trained-weight artifact IDs.
Dataset versions and produced model/score IDs must be bound before execution.
"""
import argparse
import hashlib
import json
import pathlib
from decimal import Decimal
from draft_campaign_inventory import build, expand_power


def case_id(stage, axes):
    axes=dict(axes)
    for key in ('epsilon','eta','sr_fraction'):
        if key in axes:
            axes[key]=str(Decimal(str(axes[key])).normalize())
    value={'schema':1,'stage':stage,'axes':axes}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def member_identity(node, member):
    if not 0 <= member < node['member_count']:
        raise ValueError('Member outside planned ensemble')
    seeds=node.get('member_seeds',list(range(node['member_count'])))
    if len(seeds)!=node['member_count'] or len(set(seeds))!=len(seeds):
        raise ValueError('Invalid explicit member seeds')
    seed=seeds[member]
    return {'case_id':node['id'],'member':seed,'model_seed':seed,
            'training_order_seed':seed,'data_split_seed':seed,
            'ensemble_member_seed':seed,
            'smearing_noise_seed':int(node['axes']['mother_seed']) if node['stage']==2 else 0,
            'initialization_policy': 'phase2_identity_model_init_v1' if node['stage']==3 else 'per_member_model_seed_v1'}


def dependency_plan(inventory):
    powers=list(expand_power(inventory))
    bases={(r['signal'],r['epsilon'],r['mother_seed']) for r in powers}
    smeared={(r['signal'],r['epsilon'],r['eta'],r['mother_seed']) for r in powers if r['eta']!='inf'}
    for signal,epsilon in [('HH4b_400','0.0075'),('ZH4b','0.03')]:
        smeared.update((signal,epsilon,eta,seed) for eta in ('0.5','1.0','2.0','3.0') for seed in range(100))
    smeared.update([('HH4b','0','0.1',5),('HH4b','0.01','0.1',50)])
    assert len(bases)==inventory['counts']['base_ensembles_for_power_grid']
    assert len(smeared)==inventory['counts']['smeared_ensembles_for_power_efficiency_and_explicit_illustrations']
    nodes=[];lookup={}
    for signal,epsilon,seed in sorted(bases):
        axes={'signal':signal,'epsilon':epsilon,'mother_seed':seed}
        node={'id':case_id(1,axes),'stage':1,'axes':axes,'max_epochs':100,'member_count':15,'requires':[],
              'score_domains':['X1','X2']}
        lookup[(1,signal,epsilon,seed)]=node['id'];nodes.append(node)
    for signal,epsilon,eta,seed in sorted(smeared):
        axes={'signal':signal,'epsilon':epsilon,'eta':eta,'mother_seed':seed}
        node={'id':case_id(2,axes),'stage':2,'axes':axes,'max_epochs':30,'member_count':15,
              'requires':[lookup[(1,signal,epsilon,seed)]],'score_domains':['X1','X2'],
              'upstream_member_binding':'same member index in Step 1'}
        lookup[(2,signal,epsilon,eta,seed)]=node['id'];nodes.append(node)
    for row in powers:
        signal,epsilon,eta,seed=(row[k] for k in ('signal','epsilon','eta','mother_seed'))
        axes=dict(row)
        node={'id':case_id(3,axes),'stage':3,'axes':axes,'max_epochs':100,
              'member_count':inventory['ensemble_members']['step3_initial'],
              'requires':([lookup[(1,signal,epsilon,seed)]] if eta=='inf' else
                          [lookup[(1,signal,epsilon,seed)],lookup[(2,signal,epsilon,eta,seed)]]),
              'score_domains':['X2'],
              'region_member_seeds':list(range(15)),
              'region_binding':('X1 maximum over 15 base members' if eta=='inf' else
                                'X1 maximum over 15 paired base/smeared members'),
              'aggregation':'deferred; retain all member log ratios'}
        nodes.append(node)
    diagnostic=inventory['diagnostic_scope']['original_vs_representation']
    signal,epsilon,eta,mother=(diagnostic[k] for k in ('signal','epsilon','eta','mother_seed'))
    if diagnostic['smearing_noise_seed']!=mother:
        raise ValueError('Diagnostic needs a distinct noise recipe; do not silently reuse the main upstream')
    for model in diagnostic['models']:
        axes={'signal':signal,'epsilon':epsilon,'eta':eta,'mother_seed':mother,
              'sr_fraction':diagnostic['sr_fraction'],'region_member_seed':diagnostic['upstream_model_seed'],
              'model':model['architecture']}
        nodes.append({'id':case_id(3,axes),'stage':3,'axes':axes,'member_count':1,
            'member_seeds':[model['member_seed']],'max_epochs':model['epochs'],'depth':model['depth'],
            'requires':[lookup[(1,signal,epsilon,mother)],lookup[(2,signal,epsilon,eta,mother)]],
            'score_domains':['X2'],'purpose':'original_vs_representation',
            'region_binding':'X1 thresholds from the single paired upstream member',
            'upstream_member_seed':diagnostic['upstream_model_seed'],
            'region_member_seeds':[diagnostic['upstream_model_seed']],
            'input_space':'base_encoder' if model['architecture']=='AttentionClassifier' else 'raw'})
    by_id={node['id']:node for node in nodes}
    bindings={}
    def upstream_ids(case,eta):
        signal,epsilon,mother=(case[k] for k in ('signal','epsilon','mother_seed'))
        result=[lookup[(1,signal,epsilon,mother)]]
        if eta!='inf':result.append(lookup[(2,signal,epsilon,eta,mother)])
        return result
    def cr_node(case,eta,purpose):
        axes={key:case[key] for key in ('signal','epsilon','mother_seed','sr_fraction')}
        axes['eta']=eta;key=case_id(3,axes)
        if key not in by_id:
            node={'id':key,'stage':3,'axes':axes,'max_epochs':100,'member_count':inventory['ensemble_members']['step3_initial'],
                  'requires':upstream_ids(case,eta),'score_domains':['X2'],'purpose':purpose,
                  'region_member_seeds':list(range(15)),
                  'region_binding':'X1 maximum over 15 base members' if eta=='inf' else 'X1 maximum over 15 paired base/smeared members',
                  'aggregation':'deferred; retain all member log ratios'}
            by_id[key]=node;nodes.append(node)
        by_id[key].setdefault('used_by',[]).append(purpose)
        return key
    pull=inventory['diagnostic_scope']['null_pull']
    pull_nodes=[]
    for eta,mothers in pull['mother_seeds_by_eta'].items():
        for mother in mothers:
            pull_nodes.append(cr_node({**pull,'mother_seed':mother},eta,'null_pull'))
    bindings['null_pull']={'training_nodes':pull_nodes,'binning':pull['binning'],'nbins':pull['nbins']}
    overlap=inventory['diagnostic_scope']['null_overlap']
    bindings['null_overlap']={'upstream_nodes_by_eta':{eta:upstream_ids(overlap,eta) for eta in overlap['etas']},
                             'needs_CR_model':False}
    bindings['base_CR_histogram']={'training_nodes':[cr_node(overlap,eta,'base_CR_histogram') for eta in overlap['etas']],
                                  'needs_all_X2_CR_scores':True}
    tail=inventory['diagnostic_scope']['smearing_tail']
    bindings['smearing_tail']={'upstream_nodes_by_eta':{eta:upstream_ids(tail,eta) for eta in tail['etas']},
                              'needs_CR_model':False,'sr_fraction':tail['sr_fraction']}
    classifier=inventory['diagnostic_scope']['classifier']
    bindings['classifier']={'upstream_nodes':upstream_ids(classifier,classifier['eta']),
                            'member_seeds':classifier['selected_member_seeds'],'needs_CR_model':False}
    bindings['original_vs_representation']={'training_nodes':[n['id'] for n in nodes if n.get('purpose')=='original_vs_representation']}
    stages={n['id']:n['stage'] for n in nodes}
    assert len(stages)==len(nodes)
    assert all(stages[parent]<node['stage'] for node in nodes for parent in node['requires'])
    counts={str(stage):{'ensembles':sum(n['stage']==stage for n in nodes),
                        'networks':sum(n['member_count'] for n in nodes if n['stage']==stage)} for stage in (1,2,3)}
    assert sum(n['member_count'] for n in nodes if n['stage']==3 and n.get('purpose') is None)==inventory['counts']['initial_power_cr_networks']
    return {'schema':2,'smearing_seed_rule':'Step 2 uses mother_sample_seed; Steps 1/3 apply no new smearing',
            'status':'DEPENDENCY_PLAN_NOT_LAUNCHABLE','draft':inventory['draft'],
            'default_worker_processes_per_gpu':5,'counts':counts,'nodes':nodes,'diagnostic_bindings':bindings,
            'identity_note':'Execution grouping and target ensemble size are excluded from logical case IDs; preserve existing member identities when extending.',
            'unresolved':inventory['unresolved_before_launch']+[
                'Bind source dataset versions and actual upstream artifact IDs before training.',
                'Bind plotting selectors and evaluation recipes to the retained member scores; no plotting or inference is submitted by this graph.'],
            'launch_submitted':False}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--draft',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args()
    inventory=build(args.draft)
    result=dependency_plan(inventory)
    args.out.write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({'status':result['status'],'counts':result['counts']}))


if __name__=='__main__':
    main()
