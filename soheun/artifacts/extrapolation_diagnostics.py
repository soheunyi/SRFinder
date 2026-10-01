"""Internal eta=2/infinity SR background comparison; no power-based selection."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .evaluation import RULES
from .output_files import atomic_csv
from .training_store import canonical,sha
from .train_stage import _atomic_json


def summarize_extrapolation(cases):
    groups=defaultdict(list)
    for value in cases:
        a=value['axes']
        if value.get('analysis')!='extrapolation_bias':raise ValueError('Wrong diagnostic analysis')
        groups[(a['epsilon'],a['sr_fraction'],a['eta'])].append(value)
    if not groups:raise ValueError('No extrapolation diagnostic cases')
    rows=[];profiles=[];complete=True
    for (epsilon,size,eta),group in sorted(groups.items(),key=lambda kv:tuple(map(float,kv[0]))):
        seeds=sorted(v['axes']['mother_seed'] for v in group)
        if len(set(seeds))!=len(seeds):raise ValueError('Duplicate extrapolation mother seed')
        n=len(group);complete&=seeds==list(range(100))
        member_sets={tuple(v['member_seeds']) for v in group}
        if len(member_sets)!=1:raise ValueError('Extrapolation member sets differ')
        for rule in RULES:
            metrics=[v['rule_metrics'][rule] for v in group]
            target='all_4b' if float(epsilon)==0 else 'background_4b'
            if any(m['target']!=target or m['nbins']!=64 for m in metrics):
                raise ValueError('Extrapolation target or bin count differs')
            errors=np.asarray([m['relative_count_error'] for m in metrics])
            if not np.isfinite(errors).all():raise ValueError('Nonfinite extrapolation count error')
            pulls=np.asarray([[np.nan if x is None else x for x in m['pull']] for m in metrics])
            means=[];sds=[]
            for column in pulls.T:
                valid=np.isfinite(column).all()
                means.append(float(column.mean()) if valid else None)
                sds.append(float(column.std(ddof=1)) if valid and n>1 else None)
            base={'epsilon':epsilon,'sr_fraction':size,'eta':eta,'rule':rule,'n':n,'target':target}
            rows.append({**base,'mean_count_error':float(errors.mean()),
                'sd_count_error':float(errors.std(ddof=1)) if n>1 else None,
                'undefined_pull_bins':sum(x is None for x in means),
                'mean_shape_error':float(np.mean([m['shape_error'] for m in metrics]))
                    if all(m['shape_error'] is not None for m in metrics) else None})
            profiles.append({**base,'mean_pull':means,'sd_pull':sds,'mother_seeds':seeds,
                'case_ids':sorted(v['case_id'] for v in group),'member_seeds':group[0]['member_seeds']})
    return {'schema':1,'placement':'INTERNAL_NOT_MANUSCRIPT',
        'coverage':'FULL_100_SEEDS_PER_CELL' if complete else 'PARTIAL_DIAGNOSTIC_COVERAGE',
        'rows':rows,'pull_profiles':profiles,
        'uncertainty':'sample SD across mother seeds; not standard error',
        'null_target':'all 4b; simulation null contains no signal',
        'signal_target':'background 4b numerator only; legacy pull_bg4b uses all 4b variance',
        'variance':'sum(all 4b physical weights squared) + sum(reweighted 3b weights squared)',
        'binning':'each case uses its own X1 SR reweighted-3b 64 quantile bins; compare bin quantile positions',
        'decision':'No aggregation choice or scientific acceptance follows automatically'}


def write_extrapolation(summary,output):
    """Four small null comparison figures, one per rule; signal profiles stay in JSON."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(output)
    atomic_csv(root/'extrapolation_bias_summary.csv',summary['rows'])
    files=['extrapolation_bias_summary.csv'];references=[]
    null=[p for p in summary['pull_profiles'] if float(p['epsilon'])==0]
    sizes=sorted({float(p['sr_fraction']) for p in null})
    if not sizes:raise ValueError('Null profiles are required for the internal figure')
    for rule in RULES:
        fig,axes=plt.subplots(1,len(sizes),figsize=(4*len(sizes),4),squeeze=False)
        for ax,size in zip(axes[0],sizes):
            selected=[p for p in null if float(p['sr_fraction'])==size and p['rule']==rule]
            if {float(p['eta']) for p in selected}!={2.,float('inf')} or len(selected)!=2:
                plt.close(fig);raise ValueError('Internal figure needs both eta values per SR size')
            for p in sorted(selected,key=lambda p:float(p['eta'])):
                y=np.array([np.nan if v is None else v for v in p['mean_pull']])
                error=np.array([np.nan if v is None else v for v in p['sd_pull']])
                ax.errorbar(np.arange(64)/64,y,yerr=error,fmt='.',capsize=2,
                            label=f"eta={p['eta']}, n={p['n']}")
                references.append({'rule':rule,'sr_fraction':size,'eta':p['eta'],'case_ids':p['case_ids']})
            ax.axhline(0,color='black',linestyle='--');ax.legend()
            ax.set(title=f'SR fraction={size:g}',xlabel='X1 SR quantile-bin position',ylabel='Normalized background difference')
        fig.suptitle(f'Internal null extrapolation comparison: {rule}');fig.tight_layout()
        name=f'null_extrapolation_pulls_{rule}.pdf';tmp=root/(name+'.partial')
        try:fig.savefig(tmp,format='pdf');tmp.replace(root/name)
        finally:plt.close(fig);tmp.unlink(missing_ok=True)
        files.append(name)
    _atomic_json(root/'figure-provenance.json',{'summary_sha256':sha(canonical(summary)),
        'placement':summary['placement'],'sources':references,'uncertainty':summary['uncertainty'],
        'undefined_bins':'kept as missing values in JSON and gaps in figures',
        'outputs':{name:sha((root/name).read_bytes()) for name in files}})
    return files
