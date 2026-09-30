"""Check draft dependency scope without importing training runtimes or launching jobs."""
import argparse,ast,pathlib,sys
root=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'phase5'))
from campaign_dependencies import dependency_plan,member_identity
from draft_campaign_inventory import build
from evaluation_bindings import bind_outputs

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--draft",type=pathlib.Path,required=True)
    args=ap.parse_args()
    inventory=build(args.draft)
    plan=dependency_plan(inventory)
    # Compare the planner's smearing seed with the existing native editor function.
    source=ast.parse((root/'run_step_2_smeared_fvt_training.py').read_text())
    fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='edit_config')
    namespace={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<native edit_config>','exec'),namespace)
    for mother in (0,1,50,99):
     native=namespace['edit_config']({'dataset':{},'smearing':{},'smeared_fvt':{}},mother,.02)
     node=next(n for n in plan['nodes'] if n['stage']==2 and n['axes']['mother_seed']==mother)
     for member in (0,1,14):
      assert member_identity(node,member)['smearing_noise_seed']==native['smearing']['seed']
    diag=[n for n in plan['nodes'] if n.get('purpose')=='original_vs_representation']
    assert len(diag)==2 and {n['axes']['model'] for n in diag}=={'FvTClassifier','AttentionClassifier'}
    assert all(member_identity(n,0)['model_seed']==1 for n in diag)
    assert diag[0]['requires']==diag[1]['requires']
    assert plan['counts']['2']['networks']==54030 and plan['counts']['3']['networks']==70007
    infinite=[n for n in plan['nodes'] if n['stage']==3 and n['axes'].get('eta')=='inf']
    assert len(infinite)==2000 and {n['axes']['mother_seed'] for n in infinite}==set(range(100))
    assert sum(n['member_count'] for n in plan['nodes'])==146537
    assert len({n['id'] for n in plan['nodes']})==19105
    assert all(n['axes'].get('eta')!='inf' for n in plan['nodes'] if n['stage']==2)
    assert inventory['figure_generators']['power']['script']=='run_files/generate_continuous_affine_power_figures.py'
    assert any(o['id']=='tab:hypothesis_testing_noise_scale' for o in inventory['evaluation_outputs'])
    assert inventory['evaluation_spec']['clipping']['lower']=='log_tau_s; no lower clipping'
    by_id={n['id']:n for n in plan['nodes']}
    assert all(len(n['requires'])==1 and by_id[n['requires'][0]]['stage']==1 for n in infinite)
    small_noise=[n for n in plan['nodes'] if n['stage']==3 and n['axes'].get('eta')=='0.1']
    assert len(small_noise)==1 and small_noise[0]['axes']['mother_seed']==5 and small_noise[0]['axes']['epsilon']=='0'
    assert not plan['diagnostic_bindings']['smearing_tail']['needs_CR_model']
    hist=plan['diagnostic_bindings']['base_CR_histogram']['training_nodes']
    pull=plan['diagnostic_bindings']['null_pull']['training_nodes']
    assert sum(key in pull for key in hist)==2
    outputs=bind_outputs(inventory,plan['nodes'],plan['diagnostic_bindings'])
    assert sum(o['kind']=='figure' for o in outputs)==19
    bound={o['id']:o for o in outputs}
    assert len(bound['power_test_grid']['required_case_ids'])==14000
    assert len(bound['tab:hypothesis_testing_noise_scale']['required_case_ids'])==8000
    assert len(bound['approved_HH4b_eta_inf_power']['required_case_ids'])==2000
    assert all(len(o['required_case_ids'])==(2800 if 'ZH4b' in o['id'] else 2000)
               for o in outputs if o['id'].startswith('power_plot_'))
    assert not plan['launch_submitted'] and plan['status']=='DEPENDENCY_PLAN_NOT_LAUNCHABLE'
    node=next(n for n in plan['nodes'] if n['stage']==3 and n['member_count']==5)
    expanded={**node,'member_count':15}
    assert all(member_identity(node,i)==member_identity(expanded,i) for i in range(5))
    print('PASS: native smearing seeds, explicit diagnostic member 1, shared upstreams, distinct architectures and unchanged five-to-fifteen identities')


if __name__=="__main__":main()
