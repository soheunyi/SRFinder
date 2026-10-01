"""Pilot signal-concentration plots from retained X2 scores; no legacy writes."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from artifacts.evaluation import open_reader
from artifacts.campaign_scope import expected_cases
from artifacts.campaign_runtime import selected_nodes
from artifacts.figure_data import upstream_efficiency
from artifacts.training_store import canonical,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execution',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--tier',choices=['A','B','all'],required=True)
    args=ap.parse_args();reader,manifest=open_reader(args.execution);plan=reader.registry.plan
    nodes=selected_nodes(plan,expected_cases(plan,'pilot-'+args.tier));curves=[]
    for base in nodes:
        if base['stage']!=1 or float(base['axes']['epsilon'])==0:continue
        smooth=[n for n in nodes if n['stage']==2 and n['source_case_id']==base['id'] and float(n['axes']['eta'])==2.]
        if len(smooth)!=1:raise ValueError('Pilot needs exactly one eta=2 smeared ensemble per source')
        for eta,smear in [('inf',None),('2.0',smooth[0]['id'])]:
            for mode in ('max','mean'):
                value=upstream_efficiency(reader,base['id'],smear,ensemble_mode=mode)
                curves.append({'axes':base['axes'],'eta':eta,**value})
    if not curves:raise ValueError('No declared nonzero-signal curves')
    args.output.mkdir(parents=True,exist_ok=False)
    payload={'tier':args.tier,'training_manifest_sha256':sha(canonical(manifest)),'curves':curves}
    (args.output/'curves.json').write_text(json.dumps(payload,indent=2)+'\n')
    files=[]
    for epsilon in sorted({c['axes']['epsilon'] for c in curves},key=float):
        fig,ax=plt.subplots(figsize=(8,6))
        for eta in ('2.0','inf'):
            for mode in ('max','mean'):
                group=[c for c in curves if c['axes']['epsilon']==epsilon and c['eta']==eta and c['ensemble_mode']==mode]
                values=np.asarray([c['signal_fraction'] for c in group]);x=np.asarray(group[0]['four_b_fraction'])
                avg=values.mean(axis=0);sd=values.std(axis=0)
                line,=ax.plot(x,avg,label=f'eta={eta}, {mode} (n={len(group)})')
                ax.fill_between(x,avg-sd,avg+sd,color=line.get_color(),alpha=.12)
        ax.set(xlabel='Fraction of 4b weight selected',ylabel='Fraction of signal weight selected',
               title=f'Pilot {args.tier}: HH4b, signal ratio {epsilon}',xlim=(0,1),ylim=(0,1))
        ax.legend();fig.tight_layout();name=f'pilot-{args.tier}_HH4b_epsilon-{epsilon}_efficiency.pdf'
        fig.savefig(args.output/name);plt.close(fig);files.append(name)
    (args.output/'provenance.json').write_text(json.dumps({'curve_sha256':sha(canonical(payload)),'files':files,
        'bands':'one population SD across mother samples, not confidence intervals'},indent=2)+'\n')
    print(json.dumps({'status':'PILOT_EFFICIENCY_WRITTEN','curves':len(curves),'files':files}),flush=True)

if __name__=='__main__':main()
