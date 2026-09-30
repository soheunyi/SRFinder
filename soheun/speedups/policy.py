"""Explicit, recorded opt-in execution policy for new artifact tasks."""
from functools import wraps
import hashlib
from pathlib import Path
from .scope import execution_patches

ORDER=('nosync','fast_gbn','fast_reinforce','graphs')

def normalize(names):
    names=list(names or ())
    if len(set(names))!=len(names) or any(n not in ORDER for n in names):
        raise ValueError('Unknown or duplicate execution patch')
    if 'graphs' in names and 'fast_gbn' not in names:raise ValueError('graphs requires fast_gbn')
    return [n for n in ORDER if n in names]

def descriptor(names):
    names=normalize(names)
    root=Path(__file__).parent
    return {'patches':names,'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('patches.py','scope.py','policy.py')},
            'graph_architecture':'FvT only; attention stays eager',
            'nan_failure_timing':'epoch_boundary' if 'graphs' in names else 'original'}

def using_execution_patches(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        names=normalize(kwargs.get('execution_patches',()))
        if not names:return function(*args,**kwargs)
        if kwargs.get('device','cpu')!='cuda':raise ValueError('Execution patches require CUDA')
        kwargs['execution_patches']=names
        with execution_patches(names):
            return function(*args,**kwargs)
    return wrapped
