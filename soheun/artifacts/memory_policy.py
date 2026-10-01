"""Explicit allocation budgets; admission estimates are not GPU reservations."""

def _bytes(name,value):
    if type(value) is not int or value<0:raise ValueError(f'{name} must be nonnegative integer bytes')
    return value


def worker_budgets(free_bytes,*,workers=5,safety_bytes):
    """Coordinator takes one free-memory snapshot and assigns disjoint budgets.

    Workers must not each treat the entire free GPU as their own allocation.
    Safety margin and compute headroom must come from the workload's measured
    memory profile. This helper does not choose a scientifically different group.
    """
    _bytes('free_bytes',free_bytes);_bytes('safety_bytes',safety_bytes)
    if type(workers) is not int or workers<1:raise ValueError('Positive worker count required')
    if safety_bytes>=free_bytes:raise MemoryError('No usable GPU memory budget')
    amount=(free_bytes-safety_bytes)//workers
    if amount<1:raise MemoryError('Per-worker GPU budget is empty')
    return [amount]*workers


def choose_placement(requested,*,device,data_bytes,budget_bytes=None,headroom_bytes=None,available_bytes=None):
    """Select only data placement. Never change seeds, batches or model settings.

    headroom covers models/Adam, activations, context overhead and allocation
    fragmentation. Available memory is a second, non-reserving runtime check.
    Unexpected runtime OOM still requires recovery from a completed checkpoint.
    """
    if requested not in ('cpu','resident','auto'):raise ValueError('Unknown residency policy')
    _bytes('data_bytes',data_bytes)
    if device not in ('cpu','cuda'):raise ValueError('Unsupported compute device')
    decision={'requested':requested,'data_bytes':data_bytes,'budget_bytes':budget_bytes,
              'headroom_bytes':headroom_bytes,'available_bytes':available_bytes}
    if device=='cpu':
        if requested=='resident':raise ValueError('GPU residency requires CUDA compute')
        return {**decision,'placement':'cpu','reason':'cpu_compute'}
    if budget_bytes is None or headroom_bytes is None or available_bytes is None:
        raise ValueError('CUDA workers require an explicit budget, headroom and free-memory snapshot')
    for name,value in [('budget_bytes',budget_bytes),('headroom_bytes',headroom_bytes),('available_bytes',available_bytes)]:
        _bytes(name,value)
    limit=min(budget_bytes,available_bytes)
    if headroom_bytes>limit:raise MemoryError('Compute headroom does not fit; reduce concurrent workers or request another allocation')
    fits=data_bytes<=limit-headroom_bytes
    if requested=='resident' and not fits:
        raise MemoryError('Requested resident data exceeds the worker budget; choose auto or CPU staging')
    placement='resident' if requested!='cpu' and fits else 'cpu'
    reason='requested_cpu' if requested=='cpu' else ('data_fits_budget' if fits else 'data_exceeds_budget')
    return {**decision,'placement':placement,'reason':reason}
