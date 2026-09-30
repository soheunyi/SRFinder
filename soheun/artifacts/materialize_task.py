"""Bind one declared logical node after its upstream artifact sets complete.

The caller supplies frozen member recipes and the exact expected dataset recipe.
This function creates a task/region record, never trains or submits a job.
"""
from decimal import Decimal
from .bound_tasks import make_stage_task,resolve_source_pointer,build_stage_contexts
from .stage_completion import verify_stage_completion
from .regions import define_regions


def materialize_task(store,node,source_pointer,members,parents,*,expected_dataset):
    source=resolve_source_pointer(store,source_pointer)
    if source.hparams['dataset']!=expected_dataset:raise ValueError('Source differs from the declared dataset recipe')
    axes=node['axes'];stage=node['stage']
    if (int(expected_dataset['seed'])!=int(axes['mother_seed'])
            or Decimal(str(expected_dataset['signal_ratio']))!=Decimal(str(axes['epsilon']))):
        raise ValueError('Source mother/mixture differs from logical node')
    seeds=node.get('member_seeds',list(range(node['member_count'])))
    if len(seeds)!=node['member_count'] or len(members)!=len(seeds) or len(set(seeds))!=len(seeds):raise ValueError('Incomplete or duplicate member recipe set')
    for hp,seed in zip(members,seeds):
        if hp.get('step')!=stage or any(hp[key]!=seed for key in ('model_seed','train_seed','data_seed')):
            raise ValueError('Member seed/stage differs from logical node')
        if 'max_epochs' in node and hp['max_epochs']!=node['max_epochs']:raise ValueError('Training schedule differs from logical node')
        if 'model' in axes and hp['model']!=axes['model']:raise ValueError('Architecture differs from logical node')
        if 'depth' in node and hp['depth']!=node['depth']:raise ValueError('Depth differs from logical node')
        if stage==2 and (Decimal(str(hp['smearing']['noise_scale']))!=Decimal(str(axes['eta']))
                         or hp['smearing']['seed']!=int(axes['mother_seed'])):
            raise ValueError('Smearing recipe differs from native node recipe')
    if set(parents)!=set(node['requires']):raise ValueError('Missing or unexpected parent bindings')
    by_stage={}
    for key in node['requires']:
        receipt_id=parents[key]
        verify_stage_completion(store,receipt_id,source)
        receipt=store.read(receipt_id,'stage_completion')['identity']
        if receipt['stage'] in by_stage:raise ValueError('Ambiguous parent stage')
        models={}
        for position,model_id in enumerate(receipt['model_ids']):
            hp=store.read(model_id,'model')['identity']['training_recipe']['hparams']
            seed=int(hp['model_seed'])
            if seed in models:raise ValueError('Ambiguous parent member seed')
            models[seed]={'id':model_id,'hparams':hp,
                          'scores':{domain:values[position] for domain,values in receipt['score_ids'].items()}}
        by_stage[receipt['stage']]={'receipt':receipt,'models':models}
    upstream={}
    if stage==1:
        if by_stage:raise ValueError('Step 1 must have no parents')
    elif stage==2:
        if set(by_stage)!={1}:raise ValueError('Step 2 requires a completed Step-1 parent')
        if not set(seeds)<=set(by_stage[1]['models']):raise ValueError('Required encoder member is missing')
        upstream={'encoder_ids':[by_stage[1]['models'][seed]['id'] for seed in seeds]}
    elif stage==3:
        finite=axes['eta']!='inf'
        if set(by_stage)!=({1,2} if finite else {1}):raise ValueError('Region parents differ from declared eta')
        region_seeds=node['region_member_seeds']
        if not region_seeds or len(set(region_seeds))!=len(region_seeds):raise ValueError('Region members must be explicit and distinct')
        if any(not set(region_seeds)<=set(parent['models']) for parent in by_stage.values()):
            raise ValueError('Required region member is missing')
        base=[by_stage[1]['models'][seed] for seed in region_seeds]
        smooth=[by_stage[2]['models'][seed] for seed in region_seeds] if finite else None
        if smooth:
            for model in smooth:
                recipe=model['hparams']['smearing']
                if (Decimal(str(recipe['noise_scale']))!=Decimal(str(axes['eta']))
                        or recipe['seed']!=int(axes['mother_seed'])):
                    raise ValueError('Parent smearing recipe differs from declared region')
        fraction=Decimal(str(axes['sr_fraction']))
        region=define_regions(store,[m['scores']['X1'] for m in base],by_stage[1]['receipt']['event_metadata_ids']['X1'],
            [m['scores']['X1'] for m in smooth] if smooth else None,
            sr_fraction=float(fraction),cr_fraction=float(1-fraction))
        upstream={'region_id':region,'base_X2_scores':[m['scores']['X2'] for m in base]}
        if smooth:upstream['smeared_X2_scores']=[m['scores']['X2'] for m in smooth]
        if members[0]['model']=='AttentionClassifier':
            if len(base)!=1:raise ValueError('Representation diagnostic requires one region encoder')
            upstream['representation_encoder_id']=base[0]['id']
    else:raise ValueError('Unsupported stage')
    task=make_stage_task(source_pointer,members,upstream=upstream)
    task['logical_case_id']=node['id']
    build_stage_contexts(store,task)
    return task
