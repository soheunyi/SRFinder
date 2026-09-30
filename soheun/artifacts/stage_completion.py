"""Verify a stage's permanent weights, histories and required member scores.

No checkpoint or legacy artifact is deleted. A stage receipt is not a campaign
readiness decision and does not certify statistical calibration.
"""
import math
import numpy as np
from .training_store import canonical,sha


def put_member_history(store,model_id,epochs):
    """Small immutable per-member validation/LR trace; no group-position identity."""
    store.read(model_id,'model')
    identity={'model_id':model_id,'history_sha256':sha(canonical(epochs))}
    key=store._record('training_history',identity,epochs)
    verify_member_history(store,key,model_id)
    return key


def verify_member_history(store,key,model_id):
    record=store.read(key,'training_history')
    rows=record['payload']
    if record['identity']!={'model_id':model_id,'history_sha256':sha(canonical(rows))}:
        raise ValueError('Training history content or model differs')
    model=store.read(model_id,'model')
    recipe=model['identity']['training_recipe']
    epochs=int(recipe['hparams']['max_epochs'])
    if len(rows)!=epochs or [r['epoch'] for r in rows]!=list(range(epochs)):
        raise ValueError('Training history is incomplete or unordered')
    for row in rows:
        if not math.isfinite(row['val_loss']) or not math.isfinite(row['lr']) or row['lr']<=0 or row['batch_size']<1:
            raise ValueError('Invalid validation/LR trace')
    best=min(range(epochs),key=lambda i:rows[i]['val_loss'])
    if recipe['best_epoch']!=best or recipe['best_val_loss']!=rows[best]['val_loss']:
        raise ValueError('Best-weight selection disagrees with its validation history')
    return rows


def complete_stage(store,training,source,evaluation_splits,score_ids,event_metadata_ids):
    """Verify every required member/domain, then write a content-bound receipt.

    source is a verified source context; evaluation_splits maps domain to the
    registered split ID, score_ids maps domain to the ordered member score IDs.
    Steps 1/2 require both X1 and X2; Step 3 requires all X2 predictions.
    """
    if training.get('status')!='TRAINING_COMPLETE_EXPORT_PENDING':
        raise ValueError('Training has not completed')
    models=list(training['model_ids']);histories=list(training['history_ids'])
    if not models or len(set(models))!=len(models) or len(histories)!=len(models):
        raise ValueError('Missing or duplicated members/histories')
    records=[store.read(key,'model') for key in models]
    stages={r['identity']['training_recipe']['hparams']['step'] for r in records}
    if len(stages)!=1 or next(iter(stages)) not in (1,2,3):raise ValueError('Mixed or invalid stages')
    stage=next(iter(stages))
    required={'X1','X2'} if stage in (1,2) else {'X2'}
    if set(evaluation_splits)!=required or set(score_ids)!=required or set(event_metadata_ids)!=required:
        raise ValueError('Required evaluation domains are missing or unexpected')
    for key,hkey,record in zip(models,histories,records):
        store.payload_path(key)
        hp=record['identity']['training_recipe']['hparams']
        if hp['source_dataset_id']!=source.dataset_id or hp['max_epochs']!=training['completed_epochs']:
            raise ValueError('Model source or completed schedule differs')
        verify_member_history(store,hkey,key)
        for split_id in record['identity']['split_ids']:
            if store.read(split_id,'split')['identity']['dataset_id']!=source.dataset_id:
                raise ValueError('Training split source differs')
    for domain in sorted(required):
        split_id=evaluation_splits[domain]
        split=store.read(split_id,'split')
        if split['identity']['name']!=domain or split['identity']['dataset_id']!=source.dataset_id:
            raise ValueError('Evaluation split domain/source differs')
        rows=source.indices(domain)
        store.verify_split(split_id,rows)
        event_id=event_metadata_ids[domain]
        event_record=store.read(event_id,'event_metadata')
        if (event_record['identity'].get('dataset_id')!=source.dataset_id
                or event_record['identity'].get('split_id')!=split_id):
            raise ValueError('Physical event metadata source or ordering differs')
        events=store.load_array(event_id,'event_metadata')
        required_fields={'pool','pool_row','is_4b','weight','is_signal'}
        if events.shape!=(len(rows),) or not required_fields<=set(events.dtype.names or ()):
            raise ValueError('Physical event metadata is incomplete')
        if not np.isfinite(events['weight']).all() or not np.isin(events['is_4b'],[0,1]).all():
            raise ValueError('Invalid physical event metadata')
        expected=source.full_source[rows].to_dataset_info()
        if not np.array_equal(events['pool'],expected.file_idx) or not np.array_equal(events['pool_row'],expected.inner_idx):
            raise ValueError('Physical event IDs differ from reconstructed source order')
        if len(score_ids[domain])!=len(models):raise ValueError('Missing member predictions')
        for model_id,key in zip(models,score_ids[domain]):
            score=store.read(key,'scores')['identity']
            if (score['owner_id']!=model_id or score['split_id']!=split_id
                    or score['representation']!='log_density_ratio'):
                raise ValueError('Score owner, ordering or representation differs')
            values=store.load_array(key,'scores')
            if values.dtype!=np.float32 or values.shape!=(len(rows),) or not np.isfinite(values).all():
                raise ValueError('Invalid member predictions')
    identity={'status':'STAGE_ARTIFACTS_COMPLETE','stage':stage,'dataset_id':source.dataset_id,
              'completed_epochs':training['completed_epochs'],'model_ids':models,'history_ids':histories,
              'evaluation_splits':evaluation_splits,'score_ids':score_ids,'event_metadata_ids':event_metadata_ids,
              'payload_checksums':{key:store.read(key)['payload']['sha256']
                 for key in [*models,*event_metadata_ids.values(),*[s for keys in score_ids.values() for s in keys]]}}
    return store._record('stage_completion',identity)


def verify_stage_completion(store,key,source):
    identity=store.read(key,'stage_completion')['identity']
    training={**identity,'status':'TRAINING_COMPLETE_EXPORT_PENDING'}
    if complete_stage(store,training,source,identity['evaluation_splits'],identity['score_ids'],identity['event_metadata_ids'])!=key:
        raise ValueError('Stage completion identity differs')
    return True
