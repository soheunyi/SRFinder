"""Prune only redundant working weights/checkpoints of a verified complete case."""
from pathlib import Path
import hashlib,json
from .training_store import canonical,sha
from .case_registry import CaseRegistry
from .train_stage import _owned_run,_atomic_json


def _digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prune_completed_case(store, execution, case_id, *, apply=False):
    root=Path(execution).resolve()
    manifest=json.loads((root/'training-plan.json').read_text())
    plan=json.loads((root/'frozen-plan.json').read_text())
    if sha(canonical(plan))!=manifest['plan_sha256']:
        raise ValueError('Execution plan checksum differs')
    if str(store.root.resolve())!=manifest['store'] or case_id not in manifest['case_ids']:
        raise ValueError('Wrong store or undeclared execution case')
    with _owned_run(root,manifest,True):
        registry=CaseRegistry(store,plan,root/'registry',resume=True)
        value=registry.get(case_id)
        if value is None:raise ValueError('Case has no verified complete result')
        task=root/'tasks'/case_id
        if task.is_symlink() or task.resolve().parent!=root/'tasks':raise ValueError('Case output escapes execution')
        stage_plan=json.loads((task/'training-plan.json').read_text())
        with _owned_run(task,stage_plan,True):
            marker=json.loads((task/'stage-completion.json').read_text())
            if marker['completion_id']!=value['completion_id']:
                raise ValueError('Working output and registered receipt differ')
            training=task/'training'
            if training.is_symlink() or (training/'models').is_symlink():
                raise ValueError('Working directories must not be symlinks')
            training_plan=json.loads((training/'training-plan.json').read_text())
            with _owned_run(training,training_plan,True):
                complete=json.loads((training/'training-completion.json').read_text())
                receipt=value['completion']
                if (complete['status']!='TRAINING_COMPLETE_EXPORT_PENDING'
                        or complete['model_ids']!=receipt['model_ids']
                        or complete['history_ids']!=receipt['history_ids']
                        or complete['completed_epochs']!=receipt['completed_epochs']):
                    raise ValueError('Working training result differs from registered completion')
                candidates=[(training/'last.ckpt',None)]
                for key in receipt['model_ids']:
                    record=store.read(key,'model')
                    member=record['identity']['estimator']['context_identity']
                    candidates.append((training/'models'/(member+'_best.pt'),record['payload']['sha256']))
                    candidates.append((training/'models'/(member+'_last.pt'),None))
                selected=[]
                for path,expected in candidates:
                    if path.is_symlink():raise ValueError('Working artifact must not be a symlink')
                    if not path.exists():continue
                    if not path.is_file() or not path.resolve().is_relative_to(training.resolve()):
                        raise ValueError('Working path is outside its owned training output')
                    digest=_digest(path)
                    if expected is not None and digest!=expected:
                        raise ValueError('Working best weights differ from permanent model payload')
                    selected.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':digest})
                report={'schema':1,'case_id':case_id,'completion_id':value['completion_id'],
                        'permanent_model_ids':receipt['model_ids'],'files':selected,
                        'reclaimable_bytes':sum(row['bytes'] for row in selected),'applied':False}
                if not apply:return report
                journal=root/'cleanup';journal.mkdir(exist_ok=True)
                name=sha(canonical({'case':case_id,'completion':value['completion_id']}))+'.json'
                previous=journal/name
                if previous.exists() and not selected:
                    saved=json.loads(previous.read_text())
                    if saved.get('applied') and saved['completion_id']==value['completion_id']:return saved
                _atomic_json(previous,report)
                for row in selected:
                    path=root/row['path']
                    if _digest(path)!=row['sha256']:
                        raise ValueError('Working artifact changed during cleanup')
                    path.unlink()
                # Use a fresh verifier: no cached verification can mask missing payloads.
                checked=CaseRegistry(store,plan,root/'registry',resume=True).get(case_id)
                if checked['completion_id']!=value['completion_id']:
                    raise ValueError('Permanent receipt changed')
                report['applied']=True
                _atomic_json(previous,report)
                return report
