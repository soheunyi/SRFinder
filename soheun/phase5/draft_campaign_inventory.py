"""Draft-scoped inventory only. Never submit jobs or resolve unknown cases by sweep."""
import argparse
import ast
import hashlib
import itertools
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
REVIEWED_DRAFT_SHA256 = 'e0c6fe0106ed91be2bb11a1b65a1f2056340738bebcc47d6da13398f4b313581'
POWER = [
    {'signal': 'HH4b', 'epsilon': ['0', '0.005', '0.0075', '0.01', '0.02'],
     'eta': ['0.5', '1.0', '2.0', '3.0', 'inf']},
    {'signal': 'HH4b_400', 'epsilon': ['0.005', '0.0075', '0.01', '0.02'], 'eta': ['2.0']},
    {'signal': 'ZH4b', 'epsilon': ['0.005', '0.0075', '0.01', '0.02', '0.03', '0.05'], 'eta': ['2.0']},
]
SR = ['0.05', '0.10', '0.15', '0.20']


def expand_power(inventory=None):
    grid=POWER if inventory is None else inventory['power_grid']
    sizes=SR if inventory is None else inventory['sr_fractions']
    seeds=range(100) if inventory is None else inventory['mother_seeds']
    for family in grid:
        for epsilon, eta, sr, seed in itertools.product(family['epsilon'], family['eta'], sizes, seeds):
            yield {'signal': family['signal'], 'epsilon': epsilon, 'eta': eta,
                   'sr_fraction': sr, 'mother_seed': seed}


def live_tex(text):
    return '\n'.join(re.split(r'(?<!\\)%', line, maxsplit=1)[0] for line in text.splitlines())


def generators():
    source = ROOT/'run_files/generate_draft_figures.py'
    module = ast.parse(source.read_text())
    tasks = next(node.value for node in module.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'TASKS' for t in node.targets))
    return {ast.literal_eval(call.args[0]): {'script': ast.literal_eval(call.args[1]),
             'figures': ast.literal_eval(call.args[2])} for call in tasks.elts}


def build(draft):
    raw = draft.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REVIEWED_DRAFT_SHA256:
        raise ValueError('Draft changed since scope review; re-audit grids before regenerating the inventory')
    text = live_tex(raw.decode())
    figures = set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text))
    figures = {pathlib.Path(name).name for name in figures}
    hand_authored = {'contribution_diagram.pdf', 'workflow.png'}
    tasks = generators()
    expected = {name for task in tasks.values() for name in task['figures']}
    actual = figures - hand_authored
    missing, extra = actual - expected, expected - actual
    if missing or extra:
        raise ValueError(f'Draft/generator mismatch: missing={sorted(missing)}, extra={sorted(extra)}')
    tables = re.findall(r'\\label\{(tab:[^}]+)\}', text)
    if tables != ['tab:hypothesis_testing_noise_scale']:
        raise ValueError(f'Review changed live tables: {tables}')
    diagnostic_path=ROOT/'phase5/draft_diagnostic_scope.json'
    diagnostic_scope=json.loads(diagnostic_path.read_text())
    cases = list(expand_power())
    keys = [tuple(row.values()) for row in cases]
    assert len(keys) == len(set(keys)) == 14000
    base = {(r['signal'], r['epsilon'], r['mother_seed']) for r in cases}
    smeared = {(r['signal'], r['epsilon'], r['eta'], r['mother_seed']) for r in cases if r['eta']!='inf'}
    for signal, epsilon in [('HH4b_400', '0.0075'), ('ZH4b', '0.03')]:
        smeared.update((signal, epsilon, eta, seed) for eta in ['0.5', '1.0', '2.0', '3.0'] for seed in range(100))
    # Existing illustration inputs: finite eta=0.1 only for the two displayed cases.
    smeared.update([('HH4b', '0', '0.1', 5), ('HH4b', '0.01', '0.1', 50)])
    return {
        'schema': 1, 'status': 'INVENTORY_NOT_LAUNCHABLE',
        'draft': {'name': draft.name, 'sha256': hashlib.sha256(raw).hexdigest()},
        'scope': 'live draft plus explicitly approved HH4b eta=infinity power; no other added configurations',
        'scope_amendments': [{'decision': 'Add HH4b eta=infinity power with existing epsilon/SR/seed grid',
            'source': 'https://github.com/soheunyi/SRFinder/issues/6#issuecomment-5917695163',
            'draft_placement': 'UNDECIDED; retain results separately until user selects placement'}],
        'power_grid': POWER, 'sr_fractions': SR, 'mother_seeds': list(range(100)),
        'bootstrap_replicates': 1000, 'alpha': 0.05,
        'ensemble_members': {'step1': 15, 'step2': 15, 'step3_initial': 5, 'step3_optional_extension': 15},
        'step3_member_count_decision': 'Start with five; compare fixed single member versus five on development cases before deciding an extension',
        'step3_aggregation': 'UNDECIDED: mean_probability / mean_log_density_ratio / mean_density_ratio',
        'retain_all_member_log_ratios': True,
        'diagnostic_scope':diagnostic_scope,
        'diagnostic_scope_sha256':hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
        'smearing_seed_rule':'mother_sample_seed',
        'region_recipe':{'version':'sr_quantile_cr_complement_v2',
                         'complement_CR':'all events with log_psi < log_tau_s; no lower bound',
                         'partial_CR':'finite legacy quantile cutoff',
                         'source':'https://github.com/soheunyi/SRFinder/issues/6#issuecomment-5918276549'},
        'counts': {'unique_power_cr_configurations': len(cases),
                   'initial_power_cr_networks': len(cases)*5,
                   'power_cr_networks_if_extended_to_15': len(cases)*15,
                   'base_ensembles_for_power_grid': len(base),
                   'base_networks_for_power_grid': len(base)*15,
                   'smeared_ensembles_for_power_efficiency_and_explicit_illustrations': len(smeared),
                   'smeared_networks_for_those_cases': len(smeared)*15,
                   'original_vs_representation_cr_networks':2,
                   'additional_null_diagnostic_cr_configurations':1,
                   'additional_null_diagnostic_cr_networks':5},
        'count_limits': 'Training-scope estimate includes the verified null controls and representation diagnostic; source/output bindings and evaluation tasks remain separate.',
        'figure_generators': tasks,
        'generator_sha256': {task['script']: hashlib.sha256((ROOT/task['script']).read_bytes()).hexdigest()
                             for task in tasks.values()}, 'hand_authored': sorted(hand_authored),
        'live_tables': tables,
        'evaluation_spec': json.loads((ROOT/'phase5/evaluation_spec.json').read_text()),
        'pilot_scope': {'signal':'HH4b','epsilon':['0','0.005','0.0075','0.01','0.02'],
                        'eta':['2.0','inf'],'sr_fraction':'0.20',
                        'tier_A_seeds':list(range(10)),'tier_B_seeds':list(range(10,100)),
                        'prerequisites':['final recipe and plan','concurrency/restart acceptance','launch/recovery/evaluation commands'],
                        'status':'PREPARED_NOT_SUBMITTED',
                        'full_campaign':'requires separate user launch instruction'},
        'evaluation_outputs': [
            {'id': 'power_test_grid', 'kind': 'test_results', 'cases': len(cases),
             'scope': 'all declared power cells, including approved HH4b eta=infinity'},
            {'id': 'tab:hypothesis_testing_noise_scale', 'kind': 'table',
             'source_generator': 'run_files/aggregate_continuous_affine_full.py',
             'historical_summary': 'data/refit_bootstrap/continuous_affine_full_v1/supplementary_noise_scale_summary.csv',
             'outputs': ['supplementary_noise_scale_summary.csv', 'supplementary_noise_scale_cells.tex'],
             'signal': 'HH4b', 'eta': ['0.5','1.0','2.0','3.0'],
             'note': 'Existing live table; new outputs must use the new campaign result root. Eta=infinity placement remains a user decision.'},
            {'id': 'approved_HH4b_eta_inf_power', 'kind': 'evaluation_summary',
             'signal': 'HH4b', 'eta': ['inf'], 'cases': 2000,
             'outputs': ['HH4b_eta_inf_power_summary.csv'], 'draft_placement': 'UNDECIDED'}],
        'evaluation_generator_sha256': {'run_files/aggregate_continuous_affine_full.py':
            hashlib.sha256((ROOT/'run_files/aggregate_continuous_affine_full.py').read_bytes()).hexdigest()},
        'shared_null': 'HH4b epsilon=0 is reused across signal labels, with matching eta/SR/seed',
        'infinity_baseline': 'No Step-2 smearing training for eta=infinity',
        'unresolved_before_launch': [
            'Resolve exact metadata-selected illustration configurations, member/seed sets and evaluation row sets.',
            'Map new upstream model IDs through Step 2 and Step 3; freeze the Step-3 ensemble decision.',
            'Finish full-epoch validation and Step-2/CR engineering measurements.',
            'Agree storage layout with user; complete all-member score export and storage budget.',
            'Audit current draft inference recipe versus historical plotting sources.',
            'User explicitly starts campaign; inventory generation does not authorize submission.',
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--draft', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    inventory = build(args.draft)
    args.out.write_text(json.dumps(inventory, indent=2)+'\n')
    print(json.dumps({'status': inventory['status'], 'code_generated_figures': sum(len(t['figures']) for t in inventory['figure_generators'].values()),
                      'counts': inventory['counts']}))


if __name__ == '__main__':
    main()
