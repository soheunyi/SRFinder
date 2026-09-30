"""Bind logical cases to verified stage results without launching work."""
import json
from copy import deepcopy
from collections import OrderedDict
from decimal import Decimal
from pathlib import Path
from .training_store import canonical,sha
from .bound_tasks import build_stage_contexts,resolve_source_pointer
from .stage_completion import verify_stage_completion
from .train_stage import _owned_run
from .materialize_task import materialize_task
from .campaign_recipes import recipes
from .verification_snapshot import VerificationSnapshot


class CaseRegistry:
    def __init__(self,store,plan,root,*,resume=False):
        self.store=store;self.root=Path(root);self.plan=plan;self.plan_id=sha(canonical(plan))
        self.verified = OrderedDict()
        self.nodes={node['id']:node for node in plan['nodes']}
        if len(self.nodes)!=len(plan['nodes']):raise ValueError('Duplicate logical cases')
        if any(parent not in self.nodes or self.nodes[parent]['stage']>=node['stage']
               for node in self.nodes.values() for parent in node['requires']):
            raise ValueError('Registry plan has invalid or cyclic dependencies')
        self.manifest={'schema':1,'plan_sha256':self.plan_id}
        with _owned_run(self.root,self.manifest,resume):
            (self.root/'cases').mkdir(exist_ok=True)

    def _path(self,case_id):
        if case_id not in self.nodes:raise ValueError('Case is outside the declared plan')
        return self.root/'cases'/(sha(canonical(case_id))+'.json')

    def _verify(self,case_id,task,completion_id):
        node=self.nodes[case_id]
        if task.get('logical_case_id')!=case_id or task['stage']!=node['stage']:
            raise ValueError('Task is not bound to the declared case')
        if len(task['members'])!=node['member_count']:raise ValueError('Incomplete declared member set')
        expected_seeds=node.get('member_seeds',list(range(node['member_count'])))
        if [hp['model_seed'] for hp in task['members']]!=expected_seeds:
            raise ValueError('Task member seeds differ from plan')
        for hp,seed in zip(task['members'],expected_seeds):
            if any(hp[key]!=seed for key in ('train_seed','data_seed')):raise ValueError('Task data/shuffle seeds differ from plan')
            if 'depth' in node and hp['depth']!=node['depth']:raise ValueError('Task depth differs from plan')
            if 'max_epochs' in node and hp['max_epochs']!=node['max_epochs']:
                raise ValueError('Task schedule differs from plan')
        contexts,source=build_stage_contexts(self.store,task)
        dataset=source.hparams['dataset']
        if (int(dataset['seed'])!=node['axes']['mother_seed']
                or Decimal(str(dataset['signal_ratio']))!=Decimal(str(node['axes']['epsilon']))):
            raise ValueError('Task source axes differ from plan')
        if 'sources' in self.plan:
            if task['source']!=self.plan['sources'][node['source_case_id']]['source']:
                raise ValueError('Task source binding differs from frozen plan')
            if canonical(task['members']) != canonical(recipes(self.plan,node)):
                raise ValueError('Task optimizer or member recipe differs from frozen plan')
        parent_results=self.parents(case_id)
        if parent_results is None:raise ValueError('A declared parent has no verified result')
        expected_task=materialize_task(self.store,node,task['source'],task['members'],parent_results,expected_dataset=dataset)
        if expected_task!=task:raise ValueError('Task upstreams differ from registered parents')
        verify_stage_completion(self.store,completion_id,source)
        completion=self.store.read(completion_id,'stage_completion')['identity']
        if completion['stage']!=node['stage'] or len(completion['model_ids'])!=len(contexts):
            raise ValueError('Completed stage/member count differs from task')
        expected={context.hash:{k:v for k,v in context.hparams.items() if not k.startswith('aux_info')}
                  for context in contexts}
        actual={}
        for key in completion['model_ids']:
            record=self.store.read(key,'model')['identity']
            context_id=record['estimator']['context_identity']
            if context_id in actual:raise ValueError('Duplicate completed estimator identity')
            actual[context_id]=record['training_recipe']['hparams']
        if actual!=expected:raise ValueError('Completed models were trained with a different task recipe')
        return completion,source

    def publish(self,case_id,task,completion_id):
        path=self._path(case_id)
        with _owned_run(self.root,self.manifest,True):
            self._verify(case_id,task,completion_id)
            task_id=self.store._record('stage_task',{'task':task})
            result_id=self.store._record('case_result',{'plan_sha256':self.plan_id,'case_id':case_id,
                'task_id':task_id,'stage_completion_id':completion_id})
            # A case cannot silently acquire a second result under this plan.
            self.store._publish(path,canonical({'schema':1,'plan_sha256':self.plan_id,
                                               'case_id':case_id,'result_id':result_id}))
        return result_id

    def get(self,case_id):
        path=self._path(case_id)
        if case_id in self.verified:
            value,snapshot=self.verified[case_id]
            snapshot.verify_unchanged()
            self.verified.move_to_end(case_id)
            return deepcopy(value)
        if not path.exists():return None
        index=json.loads(path.read_text())
        if index.get('plan_sha256')!=self.plan_id or index.get('case_id')!=case_id:
            raise ValueError('Case index does not belong to this plan')
        result=self.store.read(index['result_id'],'case_result')['identity']
        if result['plan_sha256']!=self.plan_id or result['case_id']!=case_id:
            raise ValueError('Case result identity differs from index')
        task=self.store.read(result['task_id'],'stage_task')['identity']['task']
        source=resolve_source_pointer(self.store,task['source'])
        if self.parents(case_id) is None:raise ValueError('A declared parent has no verified result')
        external=[path,self.root/'training-plan.json',task['source']['mother_record'],*source.full_source.files]
        # Capture before full verification, then reject changes during the check.
        for parent in self.nodes[case_id]['requires']:
            external.extend(self.verified[parent][1].stamps)
        snapshot=VerificationSnapshot(self.store,index['result_id'],external)
        completion,_=self._verify(case_id,task,result['stage_completion_id'])
        snapshot.verify_unchanged()
        value={'result_id':index['result_id'],'task':task,
               'completion_id':result['stage_completion_id'],'completion':completion}
        self.verified[case_id]=(deepcopy(value),snapshot)
        # Bound metadata memory too; no score arrays or context tensors are retained.
        if len(self.verified)>64:self.verified.popitem(last=False)
        return value

    def parents(self,case_id):
        self._path(case_id)
        result={}
        for parent in self.nodes[case_id]['requires']:
            value=self.get(parent)
            if value is None:return None
            result[parent]=value['completion_id']
        return result
