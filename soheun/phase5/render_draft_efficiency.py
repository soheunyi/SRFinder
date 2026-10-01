"""Render all eight declared efficiency figures from completed new-store cases."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from artifacts.evaluation import open_reader
from artifacts.figure_data import upstream_efficiency,model_aware_efficiency
from artifacts.training_store import canonical,sha


def render(reader,manifest,output,selected=None):
    plan=reader.registry.plan
    bindings=[b for b in plan['evaluation_bindings'] if b.get('generator','').endswith('/signal_concentration.py')]
    if selected:
        if set(selected)-{b['id'] for b in bindings}:raise ValueError('Figure is not a declared efficiency output')
        bindings=[b for b in bindings if b['id'] in selected]
    if not bindings:raise ValueError('No declared efficiency figures')
    for b in bindings:
        if Path(b['id']).name!=b['id']:raise ValueError('Invalid figure basename')
        for key in b['required_case_ids']:reader.case(key)
    root=Path(output);root.mkdir(parents=True,exist_ok=False);cache={};outputs=[]
    for binding in bindings:
        name=binding['id'];nodes=[reader.registry.nodes[key] for key in binding['required_case_ids']]
        bases=sorted([n for n in nodes if n['stage']==1],key=lambda n:n['axes']['mother_seed'])
        if {n['axes']['mother_seed'] for n in bases}!=set(range(100)) or len(bases)!=100:
            raise ValueError('Draft efficiency figure requires exactly mother seeds 0–99')
        if 'max_vs_mean' in name:
            eta='inf' if 'eta=inf' in name else '2.0';lines=[(eta,'max'),(eta,'mean')]
        elif 'vs_baseline' in name:lines=[('2.0','max'),('baseline','baseline')]
        else:lines=[(eta,'max') for eta in ('0.5','1.0','2.0','3.0','inf')]
        data=[];fig,ax=plt.subplots(figsize=(8,6))
        for eta,mode in lines:
            group=[]
            for base in bases:
                key=(base['id'],eta,mode)
                if key not in cache:
                    if eta=='baseline':cache[key]=model_aware_efficiency(reader,base['id'])
                    else:
                        matches=[] if eta=='inf' else [n for n in nodes if n['stage']==2 and n['source_case_id']==base['id'] and float(n['axes']['eta'])==float(eta)]
                        if eta!='inf' and len(matches)!=1:raise ValueError('Required eta-specific ensemble missing or ambiguous')
                        cache[key]=upstream_efficiency(reader,base['id'],matches[0]['id'] if matches else None,ensemble_mode=mode)
                    cache[key]['axes']=base['axes']
                group.append(cache[key])
            curves=np.asarray([r['signal_fraction'] for r in group]);x=np.asarray(group[0]['four_b_fraction'])
            label='Model-aware benchmark' if eta=='baseline' else f'eta={eta}, {mode}'
            line,=ax.plot(x,curves.mean(axis=0),label=label)
            ax.fill_between(x,curves.mean(axis=0)-curves.std(axis=0),curves.mean(axis=0)+curves.std(axis=0),alpha=.15,color=line.get_color())
            data.append({'eta':eta,'mode':mode,'curves':group})
        ax.set(xlabel='Fraction of 4b weight selected',ylabel='Fraction of signal weight selected',xlim=(0,1),ylim=(0,1),
               title=f"{bases[0]['axes']['signal']}, signal ratio {bases[0]['axes']['epsilon']}")
        ax.legend(loc='lower right');fig.tight_layout();fig.savefig(root/name);plt.close(fig)
        payload={'figure':name,'training_manifest_sha256':sha(canonical(manifest)),'required_case_ids':binding['required_case_ids'],
                 'lines':data,'bands':'one population SD across 100 mother samples'}
        (root/(name+'.json')).write_text(json.dumps(payload,indent=2)+'\n');outputs.append(name)
    (root/'provenance.json').write_text(json.dumps({'figures':outputs,'training_manifest_sha256':sha(canonical(manifest))},indent=2)+'\n')
    return outputs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execution',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--figure',action='append')
    args=ap.parse_args();torch.set_num_threads(1);reader,manifest=open_reader(args.execution)
    print(json.dumps({'figures':render(reader,manifest,args.output,args.figure)}),flush=True)

if __name__=='__main__':main()
