"""Render saved pre-power pull diagnostics into a new output directory."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from artifacts.training_store import canonical,sha
from artifacts.evaluation import RULES


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--diagnostics',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--rule',choices=RULES,required=True)
    ap.add_argument('--scope',choices=['pilot','draft'],required=True);args=ap.parse_args()
    manifest=json.loads((args.diagnostics/'training-plan.json').read_text())
    summary=json.loads((args.diagnostics/'summary.json').read_text())
    complete=json.loads((args.diagnostics/'completion.json').read_text())
    if (complete['summary_sha256']!=sha(canonical(summary)) or summary['manifest_sha256']!=sha(canonical(manifest))
            or complete['case_count']!=len(manifest['case_ids'])):raise ValueError('Incomplete or changed diagnostic run')
    groups={2.:[],float('inf'):[]}
    for key in manifest['case_ids']:
        record=json.loads((args.diagnostics/'cases'/(key+'.json')).read_text());value=record['value']
        if record['checksum']!=sha(canonical(value)) or value['case_id']!=key:raise ValueError('Corrupt diagnostic case')
        if args.scope=='draft' and value['axes']['mother_seed']>=50:continue
        groups[float(value['axes']['eta'])].append(value)
    if any(not g for g in groups.values()):raise ValueError('Both eta groups are required')
    if args.scope=='draft' and (manifest['nbins']!=64 or any({v['axes']['mother_seed'] for v in group}!=set(range(50)) for group in groups.values())):
        raise ValueError('Draft pull figure requires 64 bins and seeds 0–49 at both eta')
    fig,axes=plt.subplots(1,2,figsize=(14,4));provenance=[]
    for ax,eta in zip(axes,(float('inf'),2.)):
        group=groups[eta];raw=[v['rule_metrics'][args.rule]['pull'] for v in group]
        if any(any(p is None for p in row) for row in raw):raise ValueError('Undefined zero-variance pulls; inspect diagnostic report')
        array=np.asarray(raw,dtype=float);x=np.arange(array.shape[1])/array.shape[1]
        ax.errorbar(x,array.mean(axis=0),yerr=array.std(axis=0),fmt='o',capsize=3)
        ax.axhline(0,color='black',linestyle='--');ax.set(title=f'eta={eta}, n={len(group)}',xlabel='Quantile of learned SR score',ylabel='Normalized difference')
        provenance.append({'eta':str(eta),'case_ids':[v['case_id'] for v in group]})
    fig.tight_layout();args.output.mkdir(parents=True,exist_ok=False)
    name=('pilot_' if args.scope=='pilot' else '')+'pull_vs_sr_stats_SR_size_0.2.pdf'
    fig.savefig(args.output/name);plt.close(fig)
    (args.output/'provenance.json').write_text(json.dumps({'diagnostic_summary_sha256':sha(canonical(summary)),
        'rule':args.rule,'scope':args.scope,'sources':provenance,'error_bars':'population SD across mother samples'},indent=2)+'\n')
    print(json.dumps({'status':'PULL_FIGURE_WRITTEN','file':name}),flush=True)

if __name__=='__main__':main()
