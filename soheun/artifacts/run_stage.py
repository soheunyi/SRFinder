"""Resumable training -> required exports -> verified stage receipt.

This function does not select experiment configurations, submit jobs, run
bootstrap inference, delete checkpoints, or authorize campaign execution.
"""
from pathlib import Path
import hashlib
import json
import gc
import torch
from .train_stage import train_stage,_owned_run,_atomic_json
from .export_stage import export_stage,_profile
from .stage_completion import complete_stage,verify_stage_completion


def run_stage(store,contexts,source,output,*,device='cpu',resident=False,resume=False,
              export_batch_size=1024,stop_after_completed_epochs=None,
              device_budget_bytes=None,compute_headroom_bytes=None):
    if not contexts:raise ValueError('At least one member is required')
    hps=[{k:v for k,v in context.hparams.items() if not k.startswith('aux_info')} for context in contexts]
    if any(hp['source_dataset_id']!=source.dataset_id for hp in hps):
        raise ValueError('Training and export sources differ')
    manifest={'schema':1,'dataset_id':source.dataset_id,'contexts':[c.hash for c in contexts],
              'hparams':hps,'export_profiles':[_profile(hp,device,export_batch_size) for hp in hps],
              'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    root=Path(output)
    with _owned_run(root,manifest,resume):
        training_dir=root/'training'
        training=train_stage(store,contexts,training_dir,device=device,resident=resident,
            resume=training_dir.exists(),stop_after_completed_epochs=stop_after_completed_epochs,
            device_budget_bytes=device_budget_bytes,compute_headroom_bytes=compute_headroom_bytes)
        gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        if training['status']!='TRAINING_COMPLETE_EXPORT_PENDING':return training
        exported=export_stage(store,training['model_ids'],source,device=device,batch_size=export_batch_size)
        key=complete_stage(store,training,source,exported['evaluation_splits'],exported['score_ids'],
                           exported['event_metadata_ids'])
        verify_stage_completion(store,key,source)
        result={'status':'STAGE_ARTIFACTS_COMPLETE','completion_id':key,
                'model_ids':training['model_ids'],'history_ids':training['history_ids'],**exported}
        marker=root/'stage-completion.json'
        if marker.exists():
            if json.loads(marker.read_text())!=result:raise ValueError('Completed stage changed on resume')
        else:_atomic_json(marker,result)
        return result
