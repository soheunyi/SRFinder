"""Scoped GPU settings matching the existing validated training drivers.

Float32 storage is unchanged. Medium matmul permits reduced internal precision
on supported CUDA hardware, as in the existing stacked fit path. These settings
are process-global, so stage calls serialize within a process; concurrency uses
separate worker processes.
"""
from contextlib import contextmanager
from functools import wraps
import threading
import torch

_LOCK=threading.RLock()


def numerical_state():
    return {'matmul_precision':torch.get_float32_matmul_precision(),
            'cuda_matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32':torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'cudnn_deterministic':torch.backends.cudnn.deterministic}


@contextmanager
def runtime_policy(device):
    with _LOCK:
        saved=numerical_state()
        try:
            if device=='cuda':
                torch.set_float32_matmul_precision('medium')
                torch.backends.cudnn.allow_tf32=True
                torch.backends.cudnn.benchmark=False
            yield
        finally:
            if device=='cuda':
                # Restore the coupled matmul switches in this order so medium
                # is not accidentally converted to high by the boolean setter.
                torch.backends.cuda.matmul.allow_tf32=saved['cuda_matmul_tf32']
                torch.set_float32_matmul_precision(saved['matmul_precision'])
                torch.backends.cudnn.allow_tf32=saved['cudnn_tf32']
                torch.backends.cudnn.benchmark=saved['cudnn_benchmark']


def validated_gpu_runtime(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        with runtime_policy(kwargs.get('device','cpu')):
            return function(*args,**kwargs)
    return wrapped
