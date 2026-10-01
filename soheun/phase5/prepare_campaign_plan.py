"""Freeze draft/source/recipe inputs without running or submitting a campaign."""
from copy import deepcopy
import argparse,hashlib,json,pathlib
import yaml
from campaign_dependencies import dependency_plan
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from artifacts.campaign_recipes import recipes
from draft_campaign_inventory import build,REVIEWED_DRAFT_SHA256
from audit_campaign_sources import SIGNAL_FILES
ROOT=pathlib.Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def templates():
    files={'base':'better_fvt_training.yml','smear':'smeared_fvt_training_ensemble.yml',
           'raw_cr':'CR_fvt_training_original_features.yml','repr_cr':'CR_fvt_training_repr.yml'}
    configs={key:yaml.safe_load((ROOT/'configs'/name).read_text()) for key,name in files.items()}
    result={'base':configs['base']['base_fvt'],'smear':configs['smear']['smeared_fvt'],
            'raw_cr':configs['raw_cr']['CR_fvt'],'repr_cr':configs['repr_cr']['CR_fvt']}
    for hp in result.values():
        if hp['optimizer']['type']!='Adam' or hp['optimizer']['lr']!=.01:
            raise ValueError('Review changed optimizer recipe before planning')
        if hp.get('early_stop_patience') is not None:raise ValueError('Review early stopping before planning')
    return result,{name:digest(ROOT/'configs'/name) for name in files.values()}


def prepare(inventory,graph,bindings,source_store):
    if bindings.get('status')!='SOURCES_VERIFIED_TRAINING_NOT_STARTED':raise ValueError('Source preflight is incomplete')
    nodes=deepcopy(graph['nodes']);by_id={node['id']:node for node in nodes}
    sources={row['case_id']:row for row in bindings['cases']}
    base_ids={node['id'] for node in nodes if node['stage']==1}
    if set(sources)!=base_ids:raise ValueError('Verified source coverage differs from declared base cases')
    if len(sources)!=len(bindings['cases']):raise ValueError('Duplicate source bindings')
    def source_case(node):
        if node['stage']==1:return node['id']
        parents=[by_id[key] for key in node['requires'] if by_id[key]['stage']==1]
        if len(parents)!=1:raise ValueError('Every stage needs one explicit base-source dependency')
        return parents[0]['id']
    for node in nodes:
        node['source_case_id']=source_case(node)
        if node['stage']==3:node['region_recipe']=inventory['region_recipe']['version']
        dataset=sources[node['source_case_id']]['source']['hparams']['dataset']
        if (int(dataset['seed'])!=node['axes']['mother_seed'] or float(dataset['signal_ratio'])!=float(node['axes']['epsilon'])
                or dataset['signal_filename']!=SIGNAL_FILES[node['axes']['signal']]
                or dataset['n_3b']!=1000000 or dataset['ratio_4b']!=.5 or dataset['base_fvt_train_ratio']!=.5):
            raise ValueError('Source parameters differ from declared case')
    from evaluation_spec import verified_spec
    from evaluation_bindings import bind_outputs
    evaluation=verified_spec(inventory['evaluation_spec'])
    training_templates,template_hashes=templates()
    plan={'schema':1,'status':'PREPARED_CAMPAIGN_GATES_PENDING','launch_submitted':False,
          'draft':inventory['draft'],'sources':sources,'source_store':str(source_store.resolve()),
          'raw_pool_fingerprints':bindings['raw_pools'],'templates':training_templates,
          'template_sha256':template_hashes,'nodes':nodes,'counts':graph['counts'],
          'diagnostic_bindings':graph['diagnostic_bindings'],
          'scope_amendments':inventory.get('scope_amendments',[]),'region_recipe':inventory['region_recipe'],
          'evaluation_spec':evaluation,'evaluation_outputs':inventory['evaluation_outputs'],
          'pilot_scope':inventory['pilot_scope'],
          'evaluation_bindings':bind_outputs(inventory,nodes,graph['diagnostic_bindings']),
          'figure_generators':inventory['figure_generators'],
          'static_artifacts':inventory['static_artifacts'],'inline_figures':inventory['inline_figures'],
          'generator_sha256':{**inventory['generator_sha256'],**inventory['evaluation_generator_sha256']},'default_worker_processes_per_gpu':5,
          'step3_members_initial':5,'step3_members_optional_extension':15,
          'aggregation':inventory['step3_aggregation'],'resume_boundary':'completed_epoch',
          'fixed_settings':{'torch_version':'2.3.1.post300','dtype':'float32','learning_rate':.01,'adam_epsilon':1e-8,
                            'train_alignment':32,'retain_validation':True,'gpu_runtime':'validated_gpu_medium_v1'},
          'gates':['Native five-worker execution comparison passes; retired K100 remains an incomplete historical comparison.',
                   'Current stage APIs pass GPU and real-data end-to-end acceptance.',
                   'Production host/device memory and cache-export limits are validated.',
                   'Scientific pilot and data/checkpoint-policy comparison receive explicit disposition.',
                   'Result registry and plotting/inference readers are integrated.',
                   'User explicitly starts training; preparation does not submit jobs.']}
    consumer_files=('artifacts/figure_data.py','artifacts/illustration_data.py',
                    'phase5/render_campaign_figures.py','phase5/render_campaign_power.py',
                    'phase5/render_campaign_efficiency.py','phase5/render_draft_efficiency.py',
                    'phase5/render_campaign_pulls.py','phase5/render_campaign_illustrations.py')
    plan['figure_consumer_sha256']={name:digest(ROOT/name) for name in consumer_files}
    for node in nodes:
        hps=recipes(plan,node)
        if len(hps)!=node['member_count']:raise ValueError('Member recipe count differs')
        expected={'FvTClassifier'} if node['stage']==1 else {'AttentionClassifier'} if node['stage']==2 else {node['axes'].get('model','FvTClassifier')}
        if {hp['model'] for hp in hps}!=expected:raise ValueError('Declared architectures differ')
    return plan


def main():
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--draft',type=pathlib.Path);group.add_argument('--inventory',type=pathlib.Path)
    ap.add_argument('--source-bindings',type=pathlib.Path,required=True);ap.add_argument('--source-store',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True);args=ap.parse_args()
    inventory=build(args.draft) if args.draft else json.loads(args.inventory.read_text())
    if inventory['draft']['sha256']!=REVIEWED_DRAFT_SHA256:raise ValueError('Unreviewed draft inventory')
    for name,expected in {**inventory['generator_sha256'],**inventory['evaluation_generator_sha256']}.items():
        if digest(ROOT/name)!=expected:raise ValueError('Figure generator changed since inventory')
    graph=dependency_plan(inventory)
    plan=prepare(inventory,graph,json.loads(args.source_bindings.read_text()),args.source_store)
    plan['source_bindings_sha256']=digest(args.source_bindings)
    scripts=[pathlib.Path(__file__),ROOT/'phase5/campaign_dependencies.py',ROOT/'phase5/draft_campaign_inventory.py',ROOT/'phase5/draft_diagnostic_scope.json',ROOT/'artifacts/campaign_recipes.py']
    plan['planning_source_sha256']={str(path.relative_to(ROOT)):digest(path) for path in scripts}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as handle:json.dump(plan,handle,separators=(',',':'))
    print(json.dumps({'status':plan['status'],'counts':plan['counts'],'source_cases':len(plan['sources']),'nodes':len(plan['nodes'])}),flush=True)


if __name__=='__main__':main()
