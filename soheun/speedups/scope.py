"""Temporary process-local installation for integration tests and explicit callers.

Use only around an entire training operation. Checkpoint/model construction and
saving must occur inside the same scope; restoration prevents later tasks in a
reused worker from inheriting patches. No production default is selected here.
"""
from contextlib import contextmanager
import threading
from . import patches

_LOCK=threading.RLock()

@contextmanager
def execution_patches(names):
    import independent_training as it
    import stacked_fvt as sf
    import stacked_attention_classifier as sa
    import network_blocks as nb
    import fvt_classifier as fc
    targets=[(it,'record'),(it,'epoch_losses'),(it,'step'),(sf,'epoch_losses'),(sf,'independent_step'),
        (sf.StackedFvTClassifier,'nan_check'),(sa,'epoch_losses'),(sa,'independent_step'),
        (sa.StackedAttentionClassifier,'nan_check'),(fc.FvTClassifier,'forward'),
        (nb.GhostBatchNorm1d,'forward'),(nb.GhostBatchNorm1d,'_save_to_state_dict'),
        (nb.GhostBatchNorm1d,'_load_from_state_dict'),(nb.DijetReinforceLayer,'forward'),
        (nb.QuadjetReinforceLayer,'forward')]
    with _LOCK:
        saved=[(obj,key,getattr(obj,key)) for obj,key in targets]
        inplace=patches.GBN_INPLACE['on']
        absent=object();stats=getattr(it,'GRAPH_STATS',absent)
        try:
            patches.install(list(names))
            yield
        finally:
            for obj,key,value in saved:setattr(obj,key,value)
            patches.GBN_INPLACE['on']=inplace
            if stats is absent:
                if hasattr(it,'GRAPH_STATS'):delattr(it,'GRAPH_STATS')
            else:it.GRAPH_STATS=stats
