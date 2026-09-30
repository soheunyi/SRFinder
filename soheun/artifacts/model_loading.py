"""Load supported model architectures from immutable artifact recipes."""
import torch


def model_from_record(record):
    if record.get('kind')!='model':
        raise ValueError('Expected a model artifact record')
    recipe=record['identity']['training_recipe']
    hp=recipe.get('hparams')
    if not isinstance(hp,dict):
        raise ValueError('Model artifact must declare architecture hparams')
    policy=recipe.get('initialization_policy',{})
    if isinstance(policy,dict) and 'model_seed' in policy:
        seed=int(policy['model_seed'])
    elif int(hp.get('step',1))==3:
        from phase2.identity import identity_from_hparams
        seed=identity_from_hparams(hp).seed('model_init')
    else:
        seed=int(hp.get('model_seed',0))
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if hp.get('model')=='FvTClassifier':
            from fvt_classifier import FvTClassifier
            model=FvTClassifier(num_classes=2,dim_input_jet_features=4,
                dim_dijet_features=int(hp['dim_dijet_features']),
                dim_quadjet_features=int(hp['dim_quadjet_features']),
                run_name=record['id'],device='cpu',depth=hp['depth'],repr_norm=bool(hp.get('repr_norm',False)))
        elif hp.get('model')=='AttentionClassifier':
            from attention_classifier import AttentionClassifier
            width=hp.get('dim_quadjet_features',hp.get('dim_q'))
            if width is None:raise ValueError('Missing representation dimension')
            model=AttentionClassifier(int(width),2,record['id'],depth=int(hp['depth']))
        else:
            raise ValueError(f'Unsupported model architecture: {hp.get("model")}')
    return model.float()


def load_model(store, model_id, device='cpu'):
    record=store.read(model_id,'model')
    weights=torch.load(store.payload_path(model_id),map_location='cpu',weights_only=True)
    if not isinstance(weights,dict) or not all(isinstance(t,torch.Tensor) for t in weights.values()):
        raise ValueError('Expected best model state, not a full trainer checkpoint')
    if any(t.is_floating_point() and (t.dtype!=torch.float32 or not torch.isfinite(t).all()) for t in weights.values()):
        raise ValueError('Model weights must be finite float32')
    model=model_from_record(record)
    model.load_state_dict(weights,strict=True)
    return model.to(device).eval()
