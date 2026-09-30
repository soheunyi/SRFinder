"""Ragged minibatch execution and per-estimator loss accounting."""
import torch


def record(module, prefix, batches, values):
    active = next(b for b in batches if b is not None)
    zero = active[0].new_zeros(())
    losses = torch.stack([v.detach() if v is not None else zero for v in values]).cpu().view(1, -1)
    weights = torch.stack([b[2].sum() if b is not None else zero for b in batches]).detach().cpu().view(1, -1)
    counts = torch.tensor([[len(b[1]) if b is not None else 0 for b in batches]])
    for suffix, row in (('losses_per_stack', losses), ('weights_per_stack', weights), ('batch_sizes', counts)):
        name = prefix + '_' + suffix
        previous = getattr(module, name)
        setattr(module, name, torch.cat((previous, row), 0) if previous.numel() else row)


def epoch_losses(module, prefix):
    counts = getattr(module, prefix + '_batch_sizes')
    losses = getattr(module, prefix + '_losses_per_stack')
    weights = getattr(module, prefix + '_weights_per_stack')
    if counts.ndim == 1:
        return (losses * counts.view(-1, 1)).sum(0) / weights.sum(0)
    # Reduce only this estimator's own batches. Extra inactive rows must not
    # change its floating-point reduction tree or monitored scheduler metric.
    result = []
    for i in range(counts.shape[1]):
        active = counts[:, i] > 0
        result.append((losses[active, i] * counts[active, i]).sum() / weights[active, i].sum())
    return torch.stack(result)


def executor_for(module, members):
    from phase5.batched_execution import BatchedExecution
    if not hasattr(module, '_independent_executor'):
        module._independent_executor = BatchedExecution(members)
    return module._independent_executor


def step(module, batch, members, *, training, chunk_size=0):
    batches = [tuple(t.to(module.device) for t in b) if b is not None else None for b in batch['members']]
    if len(batches) != len(members):
        raise ValueError('Wrong number of estimator minibatches')
    if chunk_size and training:
        from phase5.batched_execution import train_independent_step
        optimizers = module.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        values = train_independent_step(executor_for(module, members), optimizers, batches,
                                        chunk_size, module.manual_backward)
    else:
        values = [None] * len(members)
        if chunk_size:
            groups = {}
            for i, b in enumerate(batches):
                if b is not None:
                    key = tuple((tuple(t.shape), t.dtype, t.device) for t in b)
                    groups.setdefault(key, []).append(i)
            from phase5.batched_execution import losses
            executor = executor_for(module, members)
            for indices in groups.values():
                for start in range(0, len(indices), chunk_size):
                    ids = indices[start:start+chunk_size]
                    x, y, w = (torch.stack([batches[i][j] for i in ids], 1) for j in range(3))
                    result = losses(executor.forward(ids, x), y, w)
                    for j, i in enumerate(ids):
                        values[i] = result[j]
        else:
            optimizers = module.optimizers() if training else None
            if training and not isinstance(optimizers, (list, tuple)):
                optimizers = [optimizers]
            for i, b in enumerate(batches):
                if b is None:
                    continue
                x, y, w = b
                if training:
                    optimizers[i].zero_grad()
                loss = (module.ce_loss(members[i](x), y) * w).mean()
                if training:
                    module.manual_backward(loss)
                    optimizers[i].step()
                values[i] = loss
    record(module, 'train' if training else 'val', batches, values)
    if training and hasattr(module.datamodule, 'note_training_batch'):
        module.datamodule.note_training_batch()
