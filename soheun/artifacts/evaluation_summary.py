"""Checked, new-root summaries of immutable affine-test result directories."""
from collections import defaultdict
import csv,json
from pathlib import Path
from scipy.stats import beta
from .training_store import canonical,sha
from .evaluation import RULES


from .campaign_scope import expected_cases


def load_results(roots):
    values={};common=None;plan=None;sources=[]
    for root in map(Path,roots):
        manifest=json.loads((root/'training-plan.json').read_text())
        complete=json.loads((root/'completion.json').read_text())
        if (complete['status']!='EVALUATION_COMPLETE'
                or complete['evaluation_manifest_sha256']!=sha(canonical(manifest))):
            raise ValueError('Evaluation completion and manifest disagree')
        shared={key:manifest[key] for key in ('plan_sha256','spec','decision','numpy_version','kernel_binary_sha256','evaluation_source_sha256')}
        if common is not None and common!=shared:raise ValueError('Cannot combine different evaluation recipes or decisions')
        common=shared
        current=json.loads((Path(manifest['training_execution'])/'frozen-plan.json').read_text())
        if sha(canonical(current))!=manifest['plan_sha256']:raise ValueError('Training plan changed')
        plan=current
        nodes={n['id']:n for n in plan['nodes']}
        expected={(key,rule) for key in manifest['case_ids'] for rule in manifest['rules']};observed=set()
        if len(set(complete['result_ids']))!=len(complete['result_ids']):raise ValueError('Duplicate result IDs')
        for key in complete['result_ids']:
            record=json.loads((root/'results'/(key+'.json')).read_text());identity=record['identity'];value=record['value']
            if key!=sha(canonical(identity)) or record['checksum']!=sha(canonical(value)):
                raise ValueError('Corrupt result record')
            if (identity['evaluation_manifest_sha256']!=sha(canonical(manifest))
                    or identity['case_id']!=value['case_id'] or identity['rule']!=value['rule']
                    or identity['completion_id']!=value['input_audit']['stage_completion_id']):
                raise ValueError('Result provenance differs')
            if value['axes']!=nodes[value['case_id']]['axes']:raise ValueError('Result axes differ from frozen case')
            r=value['result'];spec=manifest['spec']
            if (r['bootstrap_replicates']!=spec['bootstrap_replicates'] or r['alpha']!=spec['alpha']
                    or r['p_value']!=(1+r['max_exceedances'])/(1+r['bootstrap_replicates'])
                    or r['reject']!=(r['p_value']<=r['alpha'])):
                raise ValueError('Result test settings or p-value arithmetic differ')
            pair=(value['case_id'],value['rule'])
            if pair in values:raise ValueError('Overlapping result shards would double-count a case')
            values[pair]=value;observed.add(pair)
        if observed!=expected:raise ValueError('Incomplete evaluation result set')
        sources.append({'root':str(root.resolve()),'manifest_sha256':sha(canonical(manifest))})
    if common is None:raise ValueError('At least one completed evaluation is required')
    return plan,common,list(values.values()),sources


def summarize(roots,output,*,scope,rules=RULES):
    plan,recipe,values,sources=load_results(roots);expected=expected_cases(plan,scope)
    actual={(v['case_id'],v['rule']) for v in values}
    if actual!={(key,rule) for key in expected for rule in rules}:
        raise ValueError('Results do not exactly cover the requested scope and rules')
    grouped=defaultdict(list)
    for value in values:
        a=value['axes'];grouped[(a['signal'],a['epsilon'],a['eta'],a['sr_fraction'],value['rule'])].append(value)
    rows=[]
    for key,group in grouped.items():
        seeds=[v['axes']['mother_seed'] for v in group]
        if len(seeds)!=len(set(seeds)):raise ValueError('Duplicated mother seed in a summary cell')
        n=len(group);rejected=sum(bool(v['result']['reject']) for v in group)
        rows.append(dict(zip(('signal','epsilon','eta','sr_fraction','rule'),key),n=n,rejections=rejected,
            rejection_rate=rejected/n,ci_lower95=0. if rejected==0 else float(beta.ppf(.025,rejected,n-rejected+1)),
            ci_upper95=1. if rejected==n else float(beta.ppf(.975,rejected+1,n-rejected)),
            mean_p_value=sum(v['result']['p_value'] for v in group)/n,
            mean_bootstrap_s=sum(v['bootstrap_s'] for v in group)/n))
    rows.sort(key=lambda r:(r['signal'],float(r['eta']),float(r['epsilon']),float(r['sr_fraction']),r['rule']))
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    def write_csv(name,selected):
        if not selected:return
        with (root/name).open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    write_csv('all_rules_summary.csv',rows)
    primary=[r for r in rows if r['rule']==recipe['decision']['primary_rule']]
    if not primary:raise ValueError('Primary rule was not evaluated')
    power=[r for r in primary if float(r['eta'])==2.]
    table=[r for r in primary if r['signal']=='HH4b' and float(r['eta'])!=float('inf')]
    infinite=[r for r in primary if r['signal']=='HH4b' and r['eta']=='inf']
    write_csv('power_figure_summary.csv',power)
    write_csv('supplementary_noise_scale_summary.csv',table)
    write_csv('HH4b_eta_inf_power_summary.csv',infinite)
    # Separate infinity cells until the user decides manuscript placement.
    for name,selected in [('supplementary_noise_scale_cells.tex',table),('HH4b_eta_inf_table_cells.tex',infinite)]:
        with (root/name).open('x') as handle:
            handle.write('% Generated from verified new-store evaluation; scope: '+scope+'\n')
            for row in selected:
                handle.write(f"% eta={row['eta']}, epsilon={row['epsilon']}, SR={row['sr_fraction']}, n={row['n']}\n")
                handle.write(f"{row['rejection_rate']:.2f}\\\\\n")
    audit={'schema':1,'scope':scope,'plan_sha256':recipe['plan_sha256'],'case_count':len(expected),
           'rules':list(rules),'recipe':recipe,'sources':sources,'cells':len(rows),
           'outputs':{p.name:sha(p.read_bytes()) for p in root.iterdir() if p.is_file()},
           'scientific_acceptance':'User decision; a completed evaluation is not a calibration pass'}
    (root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit
