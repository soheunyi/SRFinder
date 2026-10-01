"""Resumable training -> required exports -> verified stage receipt.

This function does not select experiment configurations, submit jobs, run
bootstrap inference, delete checkpoints, or authorize campaign execution.
"""
from pathlib import Path
import hashlib
import json
import gc
import time
import torch
from .train_stage import train_stage,_owned_run,_atomic_json
from .runtime_policy import validated_gpu_runtime,numerical_state
from .export_stage import export_stage,_profile
from .stage_completion import complete_stage,verify_stage_completion
from speedups.policy import using_execution_patches,descriptor


@validated_gpu_runtime
@using_execution_patches
def run_stage(store,contexts,source,output,*,device='cpu',resident=False,resume=False,
              export_batch_size=1024,stop_after_completed_epochs=None,
              device_budget_bytes=None,compute_headroom_bytes=None,execution_patches=()):
    if not contexts:raise ValueError('At least one member is required')
    hps=[{k:v for k,v in context.hparams.items() if not k.startswith('aux_info')} for context in contexts]
    if any(hp['source_dataset_id']!=source.dataset_id for hp in hps):
        raise ValueError('Training and export sources differ')
    manifest={'schema':1,'dataset_id':source.dataset_id,'contexts':[c.hash for c in contexts],
              'hparams':hps,'export_profiles':[_profile(hp,device,export_batch_size) for hp in hps],
              'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if execution_patches:manifest['execution_policy']=descriptor(execution_patches)
    root=Path(output)
    with _owned_run(root,manifest,resume):
        if device=='cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        training_dir=root/'training'
        training=train_stage(store,contexts,training_dir,device=device,resident=resident,
            resume=training_dir.exists(),stop_after_completed_epochs=stop_after_completed_epochs,
            device_budget_bytes=device_budget_bytes,compute_headroom_bytes=compute_headroom_bytes)
        gc.collect()
        if device=='cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        trained=time.perf_counter()
        if training['status']!='TRAINING_COMPLETE_EXPORT_PENDING':return training
        exported=export_stage(store,training['model_ids'],source,device=device,batch_size=export_batch_size)
        if device=='cuda':torch.cuda.synchronize()
        exported_at=time.perf_counter()
        key=complete_stage(store,training,source,exported['evaluation_splits'],exported['score_ids'],
                           exported['event_metadata_ids'])
        verify_stage_completion(store,key,source)
        result={'status':'STAGE_ARTIFACTS_COMPLETE','completion_id':key,
                'model_ids':training['model_ids'],'history_ids':training['history_ids'],**exported}
        marker=root/'stage-completion.json'
        if marker.exists():
            if json.loads(marker.read_text())!=result:raise ValueError('Completed stage changed on resume')
        else:_atomic_json(marker,result)
        # Execution measurements are separate from immutable scientific receipts.
        metrics={'training_invocation_s':trained-started,
                 'export_invocation_s':exported_at-trained,
                 'receipt_verification_s':time.perf_counter()-exported_at,
                 'total_invocation_s':time.perf_counter()-started,
                 'training_record_preparation_s':training['preparation_s'],
                 'training_record_fit_checkpoint_s':training['fit_checkpoint_s'],
                 'training_memory_policy':training['memory_policy']}
        if execution_patches:
            import independent_training
            metrics['execution_policy']=descriptor(execution_patches)
            metrics['cuda_graphs']=dict(getattr(independent_training,'GRAPH_STATS',{}))
        if device=='cuda':
            metrics.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        _atomic_json(root/'execution-metrics.json',metrics)
        return result
