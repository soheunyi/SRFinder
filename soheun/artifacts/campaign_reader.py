"""Read explicit cases/members and derive test inputs without duplicating scores.

An aggregation rule is mandatory. Stored member scores and X1 region thresholds
remain authoritative; this reader does not run the bootstrap or change regions.
"""
import numpy as np
from .training_store import aggregate_log_ratios
from .regions import classify_X2
from .affine_inputs import prepare_affine_inputs


class CampaignReader:
    def __init__(self, registry):
        self.registry = registry
        self.store = registry.store

    def case(self, case_id):
        value = self.registry.get(case_id)
        if value is None:
            raise ValueError('Declared case has no verified completed result')
        return value

    def member_score_ids(self, case_id, domain, *, member_seeds=None):
        value = self.case(case_id)
        completion = value['completion']
        if domain not in completion['score_ids']:
            raise ValueError('This stage has no scores for that domain')
        by_seed = {}
        for model, score in zip(completion['model_ids'], completion['score_ids'][domain]):
            seed = self.store.read(model, 'model')['identity']['training_recipe']['hparams']['model_seed']
            if seed in by_seed:
                raise ValueError('Ambiguous member seed')
            by_seed[seed] = (model, score)
        seeds = (list(self.registry.nodes[case_id].get('member_seeds',
            range(self.registry.nodes[case_id]['member_count']))) if member_seeds is None else list(member_seeds))
        if not seeds or len(set(seeds)) != len(seeds) or any(seed not in by_seed for seed in seeds):
            raise ValueError('Missing or duplicate requested member seed')
        return [by_seed[seed] for seed in seeds]

    def aggregate(self, case_id, domain, *, aggregation, member_seeds=None):
        pairs = self.member_score_ids(case_id, domain, member_seeds=member_seeds)
        scores = [key for _, key in pairs]
        splits = {self.store.read(key, 'scores')['identity']['split_id'] for key in scores}
        if len(splits) != 1:
            raise ValueError('Member scores do not share event ordering')
        values = aggregate_log_ratios((self.store.load_array(key, 'scores') for key in scores), aggregation)
        return values, {'aggregation': aggregation, 'model_ids': [m for m, _ in pairs],
                        'score_ids': scores, 'split_id': next(iter(splits)), 'storage': 'derived_in_memory'}

    def held_out_regions(self, case_id):
        value = self.case(case_id)
        if value['completion']['stage'] != 3:
            raise ValueError('A CR case with frozen X1 thresholds is required')
        upstream = value['task']['upstream']
        return classify_X2(self.store, upstream['region_id'], upstream['base_X2_scores'],
                           upstream.get('smeared_X2_scores'))

    def affine_inputs(self, case_id, *, aggregation, member_seeds=None, upper=10.):
        value = self.case(case_id)
        masks = self.held_out_regions(case_id)
        log_gamma, reduction = self.aggregate(case_id, 'X2', aggregation=aggregation, member_seeds=member_seeds)
        events = self.store.load_array(value['completion']['event_metadata_ids']['X2'], 'event_metadata')
        keys = np.column_stack((events['pool'], events['pool_row'])).astype(np.int64)
        region = self.store.read(masks['region_id'], 'region')
        arrays, audit = prepare_affine_inputs(keys, events['is_4b'], events['weight'], masks['log_psi'],
            log_gamma, dataset_id=value['completion']['dataset_id'], domain='X2',
            lower=region['payload']['log_tau_s'], upper=upper)
        audit.update(case_id=case_id, stage_completion_id=value['completion_id'],
                     region_id=masks['region_id'], reduction=reduction)
        return arrays, audit
