"""Pure planning selectors for the approved power grid and pilot tiers."""

def expected_cases(plan,scope):
    powers=next((o['required_case_ids'] for o in plan.get('evaluation_bindings',[]) if o['id']=='power_test_grid'),None)
    if powers is None:raise ValueError('Plan has no explicit power-output binding')
    if scope=='full':return set(powers)
    pilot=plan['pilot_scope'];seeds=(pilot['tier_A_seeds'] if scope=='pilot-A' else
        pilot['tier_B_seeds'] if scope=='pilot-B' else pilot['tier_A_seeds']+pilot['tier_B_seeds'])
    if scope not in ('pilot-A','pilot-B','pilot-all'):raise ValueError('Unknown scope')
    powers=set(powers)
    return {n['id'] for n in plan['nodes'] if n['id'] in powers and n['axes']['signal']==pilot['signal']
            and n['axes']['epsilon'] in pilot['epsilon'] and n['axes']['eta'] in pilot['eta']
            and float(n['axes']['sr_fraction'])==float(pilot['sr_fraction']) and n['axes']['mother_seed'] in seeds}

