"""Materialize native member recipes from a frozen template/source plan."""
from copy import deepcopy


def recipes(plan, node):
    source = plan['sources'][node['source_case_id']]['source']
    stage = node['stage']
    role = ('base' if stage == 1 else 'smear' if stage == 2 else
            'repr_cr' if node.get('input_space') == 'base_encoder' else 'raw_cr')
    seeds = node.get('member_seeds', list(range(node['member_count'])))
    if len(seeds) != node['member_count'] or len(set(seeds)) != len(seeds):
        raise ValueError('Invalid explicit member seeds')
    result = []
    for seed in seeds:
        hp = deepcopy(plan['templates'][role])
        hp.update(step=stage, dataset=deepcopy(source['hparams']['dataset']),
                  model_seed=seed, train_seed=seed, data_seed=seed,
                  ensemble_member=seed, max_epochs=node['max_epochs'])
        hp['experiment_name'] = ('base_fvt_training_ensemble' if stage == 1 else
            'smeared_fvt_training_ensemble' if stage == 2 else 'CR_fvt_training_v2'
            if node.get('purpose') == 'original_vs_representation' else 'CR_fvt_training_ensemble_max')
        if 'depth' in node:
            hp['depth'] = deepcopy(node['depth'])
        if stage == 2:
            hp['smearing'] = {'noise_scale': float(node['axes']['eta']),
                'seed': int(node['axes']['mother_seed']), 'hard_cutoff': False, 'scale_mode': 'std'}
        result.append(hp)
    return result
