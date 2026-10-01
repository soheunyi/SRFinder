"""Initialize each NN from its explicit model seed, independent of group position.

Steps 1/2 match a standalone constructor under model_seed; Step 3 retains
the previously validated Phase-2 identity-derived initialization seed. It deliberately
does not infer new seeds from stack positions or timestamp-derived run names.
"""
import torch

RECIPE = 'per_member_model_seed_v1'


def initialize_members(members, hparams_list):
    if len(members) != len(hparams_list):
        raise ValueError('One initialization identity per model is required')
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    records = []
    for member, hp in zip(members, hparams_list):
        record = {'recipe': RECIPE, 'model_seed': int(hp['model_seed'])}
        if int(hp.get('step', 1)) == 3:
            # Retain the already-tested Phase-2 CR identity-to-seed mapping.
            from phase2.identity import identity_from_hparams
            identity = identity_from_hparams(hp)
            record = {'recipe': 'phase2_identity_model_init_v1',
                      'configured_model_seed': int(hp['model_seed']),
                      'model_seed': identity.seed('model_init'),
                      'estimator_identity': identity.fingerprint}
        seed = record['model_seed']
        if seed < 0 or seed >= 2**63:
            raise ValueError('model_seed must be a nonnegative 63-bit integer')
        kwargs = dict(member.hparams)
        if 'device' in kwargs:
            kwargs['device'] = torch.device('cpu')
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            reference = type(member)(**kwargs)
        member.load_state_dict(reference.state_dict(), strict=True)
        records.append(record)
    return records
