"""Export required source-domain scores with explicit feature provenance."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from constants import FEATURES
from .training_store import canonical,sha
from .runtime_policy import validated_gpu_runtime,numerical_state
from .model_loading import load_model
from .export_scores import export_member_scores


def _profile(model_hp,device,batch_size):
    root=Path(__file__).resolve().parents[1]
    files=('artifacts/runtime_policy.py','artifacts/export_stage.py','artifacts/export_scores.py','artifacts/model_loading.py',
           'artifacts/source_context.py','constants.py','fvt_classifier.py','fvt_encoder.py',
           'attention_classifier.py','network_blocks.py')
    return {'version':1,'mode':'eval','dtype':'float32','device_type':device,
            'torch_version':str(torch.__version__),'batch_size':batch_size,
            'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
            'runtime_policy':'validated_gpu_medium_v1' if device=='cuda' else 'caller_cpu',
            **numerical_state(),
            'matmul_precision':torch.get_float32_matmul_precision(),
            'cuda_matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32':torch.backends.cudnn.allow_tf32,
            'feature_layout':'contiguous_row_major','features':list(FEATURES),'transform':'base_encoder' if model_hp['model']=='AttentionClassifier' else 'raw',
            'encoder_id':model_hp.get('encoder_hash') if model_hp['model']=='AttentionClassifier' else None,
            'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}}


@validated_gpu_runtime
def export_stage(store,model_ids,source,*,device='cpu',batch_size=1024,reuse_existing=True):
    """Export all X1/X2 for Steps 1/2 or all X2 for Step 3, never aggregate.

    Physical features are shared on the host within this call. Only inference
    batches go to the device; Step-2 members use their own recorded encoders.
    """
    if device not in ('cpu','cuda') or type(batch_size) is not int or batch_size<1:
        raise ValueError('Invalid inference device or batch size')
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    if not model_ids or len(set(model_ids))!=len(model_ids):raise ValueError('Distinct members required')
    records=[store.read(key,'model') for key in model_ids]
    hps=[record['identity']['training_recipe']['hparams'] for record in records]
    stages={hp['step'] for hp in hps}
    if len(stages)!=1 or next(iter(stages)) not in (1,2,3):raise ValueError('Mixed or unsupported stages')
    stage=next(iter(stages));domains=('X1','X2') if stage in (1,2) else ('X2',)
    for key,hp in zip(model_ids,hps):
        store.payload_path(key)
        if ((stage==1 and hp['model']!='FvTClassifier') or (stage==2 and hp['model']!='AttentionClassifier')
                or (stage==3 and hp['model']=='AttentionClassifier' and hp.get('input_space')!='base_encoder')
                or hp['model'] not in ('FvTClassifier','AttentionClassifier')):
            raise ValueError('Unsupported stage architecture')
        if hp.get('source_dataset_id')!=source.dataset_id:raise ValueError('Model source differs')
        if hp['model']=='AttentionClassifier':
            encoder=hp.get('encoder_hash','')
            if not encoder.startswith('artifact:'):raise ValueError('Step 2 requires an artifact encoder')
            encoder_id=encoder.removeprefix('artifact:')
            base=store.read(encoder_id,'model')['identity']['training_recipe']['hparams']
            if base.get('step')!=1 or base.get('source_dataset_id')!=source.dataset_id:
                raise ValueError('Encoder stage/source differs')
            store.payload_path(encoder_id)
    splits={};events={};indices={};features={}
    source_hp=source.hparams['dataset']
    for domain in domains:
        rows=source.indices(domain);indices[domain]=rows
        splits[domain]=store.put_split(source.dataset_id,domain,rows,
            {'algorithm':'outer_seed_shuffle','version':1,'seed':source_hp['seed'],
             'fraction':source_hp['base_fvt_train_ratio']})
    profiles=[_profile(hp,device,batch_size) for hp in hps]
    cached={domain:[store.find_scores(key,splits[domain],'log_density_ratio',profile)
                   if reuse_existing else None for key,profile in zip(model_ids,profiles)] for domain in domains}
    for domain in domains:
        identity={'dataset_id':source.dataset_id,'split_id':splits[domain],
                  'row_order':'split_index_order','schema':'physical_events_v1'}
        event_key=sha(canonical({'schema':1,'kind':'event_metadata','identity':identity}))
        need_events=not (store.records/(event_key+'.json')).is_file()
        if need_events or any(key is None for key in cached[domain]):
            rows=indices[domain];selected=source.full_source[rows]
            frame=selected.fetch_data()
            if len(frame)!=len(rows):raise ValueError('Evaluation event count differs')
            if not np.isfinite(frame.weight.values).all() or not np.isin(frame.fourTag.values,[0,1]).all():
                raise ValueError('Invalid physical weights or labels')
            if need_events:
                info=selected.to_dataset_info()
                values=np.empty(len(rows),dtype=[('pool','<u4'),('pool_row','<i8'),('is_4b','?'),
                                                ('weight','<f8'),('is_signal','?')])
                values['pool']=info.file_idx;values['pool_row']=info.inner_idx
                values['is_4b']=frame.fourTag.values;values['weight']=frame.weight.values
                signal=np.array([Path(path).name==Path(source_hp['signal_filename']).name for path in selected.files])
                values['is_signal']=signal[info.file_idx]
                assert store.put_array('event_metadata',identity,values)==event_key
            if any(key is None for key in cached[domain]):
                tensor=torch.tensor(frame[FEATURES].values,dtype=torch.float32).contiguous()
                if not torch.isfinite(tensor).all():raise ValueError('Nonfinite evaluation features')
                features[domain]=tensor
        store.payload_path(event_key)
        events[domain]=event_key
    scores={domain:[] for domain in domains}
    for member,(key,hp,profile) in enumerate(zip(model_ids,hps,profiles)):
        encoder=None
        for domain in domains:
            if cached[domain][member] is not None:
                scores[domain].append(cached[domain][member]);continue
            if hp['model']=='AttentionClassifier' and encoder is None:
                encoder=load_model(store,hp['encoder_hash'].removeprefix('artifact:'),device=device)
            def batches():
                with torch.inference_mode():
                    for begin in range(0,len(indices[domain]),batch_size):
                        rows=indices[domain][begin:begin+batch_size]
                        x=features[domain][begin:begin+batch_size]
                        if encoder is not None:x=encoder.encoder(x.to(device)).contiguous()
                        yield rows,x
            scores[domain].append(export_member_scores(store,key,splits[domain],indices[domain],batches(),
                device=device,reuse_existing=reuse_existing,inference_recipe=profile))
    return {'evaluation_splits':splits,'event_metadata_ids':events,'score_ids':scores}
