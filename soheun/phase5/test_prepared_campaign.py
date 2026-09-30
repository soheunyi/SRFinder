"""Validate a frozen plan using existing verified source manifests; no training."""
import argparse,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from prepare_campaign_plan import recipes,prepare
from campaign_dependencies import dependency_plan

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--inventory",type=pathlib.Path,required=True)
    ap.add_argument("--source-bindings",type=pathlib.Path,required=True);args=ap.parse_args()
    inventory=json.loads(args.inventory.read_text())
    source=args.source_bindings
    bindings=json.loads(source.read_text());graph=dependency_plan(inventory)
    plan=prepare(inventory,graph,bindings,source.parent/'store')
    node=next(n for n in plan['nodes'] if n['stage']==2 and n['axes']=={'signal':'HH4b','epsilon':'0.01','eta':'2.0','mother_seed':50})
    hp=recipes(plan,node)[14]
    assert hp['model_seed']==hp['train_seed']==hp['data_seed']==14
    assert hp['smearing']=={'noise_scale':2.,'seed':50,'hard_cutoff':False,'scale_mode':'std'}
    assert hp['depth']==8 and hp['max_epochs']==30 and hp['optimizer']=={'type':'Adam','lr':.01}
    assert hp['dataloader']['batch_size_milestones']==[1,3,6,10,15]
    for node in plan['nodes']:
     if node['stage']==3 and node['axes']['eta']=='inf':assert 'smearing' not in recipes(plan,node)[0]
    diag=[n for n in plan['nodes'] if n.get('purpose')=='original_vs_representation']
    assert {recipes(plan,n)[0]['max_epochs'] for n in diag}=={30,100}
    assert all(recipes(plan,n)[0]['model_seed']==1 for n in diag)
    bad={**bindings,'cases':bindings['cases'][:-1]}
    try:prepare(inventory,graph,bad,source.parent/'store')
    except ValueError:pass
    else:raise AssertionError('Missing source silently bound')
    assert all(n['region_recipe']=='sr_quantile_cr_complement_v2' for n in plan['nodes'] if n['stage']==3)
    assert plan['evaluation_spec']['clipping']['upper']==10.0
    assert len(plan['evaluation_spec']['implementation_sha256'])==5
    assert len(plan['generator_sha256'])==len(inventory['generator_sha256'])+1
    assert not plan['launch_submitted'] and len(plan['sources'])==1500 and len(plan['nodes'])==19105
    print('PASS: complete source coverage, native recipe spot-check, diagnostic schedules and missing-source rejection')


if __name__=="__main__":main()
