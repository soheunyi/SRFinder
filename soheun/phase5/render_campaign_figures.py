"""Explicit pilot/full figure entry point using new output roots and checked inputs."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from artifacts.evaluation import open_reader,RULES
from artifacts.training_store import canonical,sha


def copy_static(plan,assets,output):
    result={}
    for name,record in plan['static_artifacts'].items():
        source=Path(assets)/name
        if hashlib.sha256(source.read_bytes()).hexdigest()!=record['sha256']:raise ValueError(f'Static artifact differs: {name}')
        shutil.copyfile(source,Path(output)/name);result[name]=record['sha256']
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execution',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--scope',choices=['draft','pilot-A','pilot-B','pilot-all'],required=True)
    ap.add_argument('--decision',type=Path,required=True);ap.add_argument('--summary',type=Path,required=True)
    ap.add_argument('--diagnostics',type=Path,required=True);ap.add_argument('--assets',type=Path)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=ap.parse_args()
    reader,manifest=open_reader(args.execution);plan=reader.registry.plan
    decision=json.loads(args.decision.read_text());rule=decision.get('primary_rule')
    if rule not in RULES or not decision.get('decision_reference'):raise ValueError('Recorded user decision required')
    summary=json.loads((args.summary/'audit.json').read_text())
    expected_scope='full' if args.scope=='draft' else args.scope
    if summary['scope']!=expected_scope or summary['recipe']['decision']!=decision:raise ValueError('Summary scope or decision differs')
    if summary['plan_sha256']!=manifest['plan_sha256']:raise ValueError('Summary belongs to another plan')
    for name,expected in plan.get('figure_consumer_sha256',{}).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Figure consumer changed since plan freeze')
    if args.scope=='draft' and args.assets is None:raise ValueError('Supply the pinned static artifact directory')
    args.output.mkdir(parents=True,exist_ok=False)
    def run(script,*extra):
        subprocess.run([sys.executable,str(ROOT/'phase5'/script),*map(str,extra)],check=True)
    run('render_campaign_power.py','--summary',args.summary,'--output',args.output/'power')
    if args.scope=='draft':
        run('render_draft_efficiency.py','--execution',args.execution,'--output',args.output/'efficiency')
        run('render_campaign_pulls.py','--diagnostics',args.diagnostics,'--output',args.output/'pulls','--rule',rule,'--scope','draft')
        run('render_campaign_illustrations.py','--execution',args.execution,'--output',args.output/'illustrations','--rule',rule,'--device',args.device)
        static=copy_static(plan,args.assets,args.output)
    else:
        run('render_campaign_efficiency.py','--execution',args.execution,'--output',args.output/'efficiency','--tier',args.scope.split('-')[1])
        run('render_campaign_pulls.py','--diagnostics',args.diagnostics,'--output',args.output/'pulls','--rule',rule,'--scope','pilot')
        static={}
    # Collect generated figures into the requested root, preserving source metadata.
    for directory in (args.output/'power',args.output/'efficiency',args.output/'pulls',args.output/'illustrations'):
        if directory.exists():
            for file in directory.glob('*.pdf'):shutil.copyfile(file,args.output/file.name)
    if args.scope=='draft':
        expected={name for task in plan['figure_generators'].values() for name in task['figures']}
        if any(not (args.output/name).is_file() for name in expected):raise ValueError('A declared draft figure is missing')
    audit={'scope':args.scope,'plan_sha256':manifest['plan_sha256'],'decision':decision,'static_artifacts':static,
           'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()}}
    (args.output/'completion.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
