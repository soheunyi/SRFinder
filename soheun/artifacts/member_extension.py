"""Append CR members under a new frozen plan; preserve all completed artifacts."""
from contextlib import contextmanager
from copy import deepcopy
import json,time
from pathlib import Path
import torch
from .training_store import canonical,sha
from .train_stage import _owned_run,_atomic_json,_implementation
from .runtime_policy import validated_gpu_runtime,numerical_state
from .case_registry import CaseRegistry
from .bound_tasks import build_stage_contexts
from .member_splits import verify_member_splits
from .stage_completion import complete_stage,verify_stage_completion,verify_member_history
from .run_stage import run_stage
from .export_stage import _profile


def make_extension_plan(origin_plan,origin_manifest,origin_execution,*,member_count,decision):
    if (type(member_count) is not int or member_count<2
            or decision.get('status')!='USER_DECISION_RECORDED'
            or decision.get('step3_member_count')!=member_count
            or not isinstance(decision.get('decision_reference'),str)
            or not decision['decision_reference'].strip()):
        raise ValueError('Member extension needs an explicit recorded count decision')
    if origin_manifest['plan_sha256']!=sha(canonical(origin_plan)):
        raise ValueError('Original plan and execution manifest differ')
    if origin_plan.get('member_extension'):
        raise ValueError('Nested extension needs a separately reviewed plan')
    plan=deepcopy(origin_plan);changed=[]
    for node in plan['nodes']:
        if node['stage']!=3 or node.get('purpose')=='original_vs_representation':continue
        if node['axes'].get('model','FvTClassifier')!='FvTClassifier':
            raise ValueError('Extension is restricted to the declared raw CR ensembles')
        seeds=node.get('member_seeds',list(range(node['member_count'])))
        if seeds!=list(range(node['member_count'])) or member_count<=len(seeds):
            raise ValueError('Extension must append to an unchanged contiguous seed prefix')
        node['member_count']=member_count;node['member_seeds']=list(range(member_count));changed.append(node['id'])
    if not changed:raise ValueError('No CR ensembles can be extended')
    plan['member_extension']={'origin_execution':str(Path(origin_execution).resolve()),
        'origin_manifest_sha256':sha(canonical(origin_manifest)),
        'origin_plan_sha256':origin_manifest['plan_sha256'],'member_count':member_count,
        'extended_case_ids':sorted(changed),'decision':deepcopy(decision),
        'decision_sha256':sha(canonical(decision))}
    plan['counts']={str(stage):{'ensembles':sum(n['stage']==stage for n in plan['nodes']),
        'networks':sum(n['member_count'] for n in plan['nodes'] if n['stage']==stage)} for stage in (1,2,3)}
    plan['step3_members_selected']=member_count
    plan['status']='PREPARED_MEMBER_EXTENSION_NOT_SUBMITTED';plan['launch_submitted']=False
    return plan


@contextmanager
def origin_guard(store,plan):
    """Hold original coordinator ownership so its completion set cannot grow."""
    meta=plan.get('member_extension')
    if meta is None:
        yield None;return
    root=Path(meta['origin_execution'])
    manifest=json.loads((root/'training-plan.json').read_text())
    original=json.loads((root/'frozen-plan.json').read_text())
    if (manifest['store']!=str(store.root.resolve())
            or sha(canonical(manifest))!=meta['origin_manifest_sha256']
            or sha(canonical(original))!=meta['origin_plan_sha256']):
        raise ValueError('Extension must reuse the exact original store, manifest and plan')
    expected=make_extension_plan(original,manifest,root,member_count=meta['member_count'],decision=meta['decision'])
    if canonical(expected)!=canonical(plan):raise ValueError('Extension changed an unrelated recipe or scope')
    with _owned_run(root,manifest,True):
        if json.loads((root/'training-plan.json').read_text())!=manifest:
            raise ValueError('Original execution changed during extension admission')
        yield CaseRegistry(store,original,root/'registry',resume=True)


def import_unchanged_cases(registry,origin,nodes):
    """Publish original upstream/fixed-diagnostic receipts under the new plan."""
    imported=[]
    for node in sorted(nodes,key=lambda n:n['stage']):
        key=node['id']
        if node!=origin.nodes[key] or registry.get(key) is not None:continue
        old=origin.get(key)
        if old is not None:
            registry.publish(key,old['task'],old['completion_id']);imported.append(key)
    return imported


def _current_numerics(device):
    result={'device':device,'cpu_threads':torch.get_num_threads(),
        'interop_threads':torch.get_num_interop_threads(),
        'runtime_policy':'validated_gpu_medium_v1' if device=='cuda' else 'caller_cpu',
        **numerical_state(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
    if device=='cuda':result.update(gpu_name=torch.cuda.get_device_name(),
        compute_capability=list(torch.cuda.get_device_capability()))
    return result


def combine_completions(store,contexts,source,original_id,added_id):
    for key in (original_id,added_id):verify_stage_completion(store,key,source)
    parts=[store.read(key,'stage_completion')['identity'] for key in (original_id,added_id)]
    if any(p['stage']!=3 for p in parts):raise ValueError('Only CR completions may be combined')
    if any(parts[0][key]!=parts[1][key] for key in ('dataset_id','completed_epochs','evaluation_splits','event_metadata_ids')):
        raise ValueError('Extension event order, physical metadata or schedule differs')
    members={}
    for part in parts:
        for i,model_id in enumerate(part['model_ids']):
            context=store.read(model_id,'model')['identity']['estimator']['context_identity']
            if context in members:raise ValueError('Duplicate extension member')
            members[context]=(model_id,part['history_ids'][i],part['score_ids']['X2'][i])
    names=[c.hash for c in contexts]
    if set(members)!=set(names):raise ValueError('Extension does not cover the declared members')
    ordered=[members[name] for name in names]
    training={'status':'TRAINING_COMPLETE_EXPORT_PENDING','completed_epochs':parts[0]['completed_epochs'],
              'model_ids':[v[0] for v in ordered],'history_ids':[v[1] for v in ordered]}
    return complete_stage(store,training,source,parts[0]['evaluation_splits'],
        {'X2':[v[2] for v in ordered]},parts[0]['event_metadata_ids'])


@validated_gpu_runtime
def append_stage_members(store,contexts,source,output,*,original_completion_id,device='cpu',
                         resident=False,resume=False,export_batch_size=1024,
                         stop_after_completed_epochs=None,device_budget_bytes=None,
                         compute_headroom_bytes=None,execution_patches=()):
    """Fit/export only absent members, then verify a complete ordered receipt."""
    if not contexts or any(c.hparams['step']!=3 for c in contexts):raise ValueError('CR contexts required')
    verify_stage_completion(store,original_completion_id,source)
    original=store.read(original_completion_id,'stage_completion')['identity']
    by_name={c.hash:c for c in contexts};kept=set()
    implementation=_implementation();numerics=_current_numerics(device)
    for model_id,history_id,score_id in zip(original['model_ids'],original['history_ids'],original['score_ids']['X2']):
        model=store.read(model_id,'model')['identity'];name=model['estimator']['context_identity']
        if name not in by_name or name in kept:raise ValueError('Original members are not a distinct target prefix')
        context=by_name[name];hp={k:v for k,v in context.hparams.items() if not k.startswith('aux_info')}
        recipe=model['training_recipe']
        if (recipe['hparams']!=hp or recipe['implementation_sha256']!=implementation
                or recipe['numerics']!=numerics):raise ValueError('Original member training recipe or numerics changed')
        if store.read(score_id,'scores')['identity'].get('inference_recipe')!=_profile(hp,device,export_batch_size):
            raise ValueError('Extension must preserve the original inference profile')
        verify_member_splits(store,context,model['split_ids']);verify_member_history(store,history_id,model_id)
        kept.add(name)
    names=[c.hash for c in contexts]
    if set(names[:len(kept)])!=kept:
        raise ValueError('Existing member order must be the unchanged target prefix')
    added=[c for c in contexts if c.hash not in kept]
    if not added:raise ValueError('No new member requested')
    manifest={'schema':1,'kind':'append_members','dataset_id':source.dataset_id,
        'contexts':names,'original_completion_id':original_completion_id,
        'added_contexts':[c.hash for c in added],'device':device,'export_batch_size':export_batch_size,
        'execution_patches':list(execution_patches)}
    root=Path(output)
    with _owned_run(root,manifest,resume):
        began=time.perf_counter()
        component=run_stage(store,added,source,root/'added',device=device,resident=resident,
            resume=(root/'added').exists(),export_batch_size=export_batch_size,
            stop_after_completed_epochs=stop_after_completed_epochs,device_budget_bytes=device_budget_bytes,
            compute_headroom_bytes=compute_headroom_bytes,execution_patches=execution_patches)
        if component['status']!='STAGE_ARTIFACTS_COMPLETE':return component
        complete=combine_completions(store,contexts,source,original_completion_id,component['completion_id'])
        extension=store._record('member_extension',{'original_completion_id':original_completion_id,
            'added_completion_id':component['completion_id'],'combined_completion_id':complete,
            'reused_model_ids':store.read(complete,'stage_completion')['identity']['model_ids'][:len(kept)],
            'added_model_ids':store.read(complete,'stage_completion')['identity']['model_ids'][len(kept):]})
        combined=store.read(complete,'stage_completion')['identity']
        result={'status':'STAGE_ARTIFACTS_COMPLETE','completion_id':complete,
            'model_ids':combined['model_ids'],'history_ids':combined['history_ids'],
            'score_ids':combined['score_ids'],'evaluation_splits':combined['evaluation_splits'],
            'event_metadata_ids':combined['event_metadata_ids'],'extension_record_id':extension}
        path=root/'stage-completion.json'
        if path.exists() and json.loads(path.read_text())!=result:raise ValueError('Completed extension changed')
        _atomic_json(path,result)
        _atomic_json(root/'execution-metrics.json',{'reused_members':len(kept),'added_members':len(added),
            'total_invocation_s':time.perf_counter()-began})
        return result


def verify_extension_record(store,key,source,combined_id):
    value=store.read(key,'member_extension')['identity']
    if value['combined_completion_id']!=combined_id:raise ValueError('Extension receipt differs')
    for part in ('original_completion_id','added_completion_id','combined_completion_id'):
        verify_stage_completion(store,value[part],source)
    old=store.read(value['original_completion_id'],'stage_completion')['identity']
    new=store.read(value['added_completion_id'],'stage_completion')['identity']
    combined=store.read(combined_id,'stage_completion')['identity']
    members={m:(h,q) for part in (old,new) for m,h,q in zip(part['model_ids'],part['history_ids'],part['score_ids']['X2'])}
    n=len(old['model_ids'])
    if (len(members)!=len(old['model_ids'])+len(new['model_ids'])
            or set(value['reused_model_ids'])!=set(old['model_ids'])
            or set(value['added_model_ids'])!=set(new['model_ids'])
            or combined['model_ids']!=value['reused_model_ids']+value['added_model_ids']
            or len(combined['model_ids'])!=len(members)
            or set(combined['model_ids'][:n])!=set(old['model_ids'])
            or any(members.get(m)!=(h,q) for m,h,q in zip(combined['model_ids'],combined['history_ids'],combined['score_ids']['X2']))):
        raise ValueError('Extension lost or changed a member artifact')
    return new
