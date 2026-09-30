"""Map every declared figure/table and power test to its training dependencies."""
import re


def bind_outputs(inventory,nodes,diagnostics):
    known={node['id'] for node in nodes}
    powers=[n for n in nodes if n['stage']==3 and n.get('purpose') is None]
    outputs=[]
    def add(name,kind,ids,**extra):
        ids=sorted(set(ids))
        if set(ids)-known:raise ValueError('Evaluation depends on undeclared training')
        outputs.append({'id':name,'kind':kind,'required_case_ids':ids,**extra})
    def upstream(values):return [key for group in values.values() for key in group]
    for task_name,task in inventory['figure_generators'].items():
        for figure in task['figures']:
            extra={'generator':task['script'],'domain':'X2'}
            if task_name=='toys':ids=[]
            elif task_name=='power':
                family='HH4b_400' if 'HH4b_400' in figure else 'ZH4b' if 'ZH4b' in figure else 'HH4b'
                ids=[n['id'] for n in powers if float(n['axes']['eta'])==2.
                     and (n['axes']['signal']==family or (n['axes']['signal']=='HH4b' and float(n['axes']['epsilon'])==0.))]
            elif task_name=='classifier':
                ids=diagnostics['classifier']['upstream_nodes'];extra['member_seeds']=diagnostics['classifier']['member_seeds']
            elif figure=='smearing_overlap_signal_ratio_0.0_seed_5.pdf':
                ids=upstream(diagnostics['null_overlap']['upstream_nodes_by_eta'])
            elif figure=='base_and_CR_fvt_scores_hist_seed_5.pdf':ids=diagnostics['base_CR_histogram']['training_nodes']
            elif figure=='smearing_and_tails_signal_ratio_0.01_seed_50.pdf':ids=upstream(diagnostics['smearing_tail']['upstream_nodes_by_eta'])
            elif task_name=='null_case':ids=diagnostics['null_pull']['training_nodes']
            elif task_name=='on_which_to_learn':ids=diagnostics['original_vs_representation']['training_nodes']
            elif task_name=='signal_concentration':
                family='HH4b_400' if 'HH4b_400' in figure else 'ZH4b' if 'ZH4b' in figure else 'HH4b'
                epsilon=float(re.search(r'_signal=([0-9.]+)_',figure).group(1))
                etas=[] if 'eta=inf' in figure else [2.] if ('eta=2.0' in figure or 'vs_baseline' in figure) else [.5,1.,2.,3.]
                ids=[n['id'] for n in nodes if n['stage']<=2 and n['axes']['signal']==family
                     and float(n['axes']['epsilon'])==epsilon
                     and (n['stage']==1 or float(n['axes']['eta']) in etas)]
            else:raise ValueError(f'No dependency binding for live figure {figure}')
            add(figure,'figure',ids,**extra)
    for output in inventory['evaluation_outputs']:
        selected=powers
        if 'signal' in output:selected=[n for n in selected if n['axes']['signal']==output['signal']]
        if 'eta' in output:selected=[n for n in selected if n['axes']['eta'] in output['eta']]
        add(output['id'],output['kind'],[n['id'] for n in selected],recipe=output)
    if len({o['id'] for o in outputs})!=len(outputs):raise ValueError('Duplicate evaluation output')
    return outputs
