"""Experimental execution batching; original modules own all persistent state.

No data selection, seeding, loss normalization, optimizer or scheduler changes.
Only homogeneous, deterministic FvT models and unmasked GhostBatchNorm are
supported. Random operations fail under vmap rather than silently sharing RNG.
"""
from __future__ import annotations

import copy
import torch
from torch.func import functional_call
from network_blocks import GhostBatchNorm1d
from fvt_classifier import FvTClassifier


class ExecutionClassifier(FvTClassifier):
    def forward(self, x):
        q = self.encoder(x)
        score = self.attention_classifier(q)
        return score, torch.isnan(q).any() | torch.isnan(score).any()


class ExecutionGhostBatchNorm(GhostBatchNorm1d):
    def forward(self, x, mask=None, debug=False):
        if mask is not None or debug:
            raise NotImplementedError("Prototype supports unmasked GBN only")
        if not self.training:
            return super().forward(x)
        batch, _, pixels = x.shape
        ngb = self.execution_ngb
        if batch % ngb:
            raise ValueError("Batch must be divisible by number of ghost batches")
        gbs = batch // ngb
        self.gbs.fill_(gbs)
        x = x.transpose(1, 2).contiguous().view(ngb, gbs * pixels, self.features, 1)
        mean = x.mean(dim=1, keepdim=True) if self.execution_center else 0
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        bm = mean.detach().mean(dim=0) if self.execution_center else 0
        bs = std.detach().mean(dim=0)
        x = (x - mean) / std
        x = x.view(batch, pixels, self.features)
        x = self.gamma * x
        x = x + self.bias
        # Mutate only the explicitly supplied per-estimator buffer slices.
        self.m.copy_(self.eta * self.m + (1 - self.eta) * bm)
        self.s.copy_(self.eta * self.s + (1 - self.eta) * bs)
        return x.transpose(1, 2)


class BatchedExecution:
    """A non-Module adapter: checkpoint names and optimizer parameters stay intact.

    A differentiable torch.stack routes gradients back to the original leaf
    parameters. Unlike stack_module_state, it does not create replacement leaves.
    Call backward before processing the next chunk to bound activation memory.
    """
    def __init__(self, models):
        if not models:
            raise ValueError("Need at least one estimator")
        self.models = list(models)
        self.template = copy.deepcopy(models[0])
        if type(self.template) is not FvTClassifier:
            raise TypeError("Prototype supports FvTClassifier only")
        self.template.__class__ = ExecutionClassifier
        reference = dict(models[0].named_modules())
        parameter_ids = set()
        for model in models:
            member_ids = {id(p) for p in model.parameters()}
            if parameter_ids.intersection(member_ids):
                raise ValueError("Independent estimators cannot share parameters")
            parameter_ids.update(member_ids)
            modules = dict(model.named_modules())
            if modules.keys() != reference.keys():
                raise ValueError("Model structures differ")
            for name, module in modules.items():
                ref = reference[name]
                if type(module) is not type(ref) or module.extra_repr() != ref.extra_repr():
                    raise ValueError(f"Model configuration differs at {name}")
                if isinstance(module, GhostBatchNorm1d) and int(module.ngb) != int(ref.ngb):
                    raise ValueError("Ghost-batch geometry differs")
                for setting in ("repr_norm", "depth", "dim_j", "dim_d", "dim_q"):
                    if getattr(module, setting, None) != getattr(ref, setting, None):
                        raise ValueError(f"Model setting {setting} differs at {name}")
            for (name, p), (other, q) in zip(models[0].named_parameters(), model.named_parameters()):
                if name != other or p.shape != q.shape or p.requires_grad != q.requires_grad:
                    raise ValueError("Parameter structures differ")
        for module in self.template.modules():
            if isinstance(module, GhostBatchNorm1d):
                module.execution_ngb = int(module.ngb)
                module.execution_center = module.bias.requires_grad
                module.__class__ = ExecutionGhostBatchNorm

    def forward(self, indices, x):
        """x has shape [batch, len(indices), features], each column its own data."""
        selected = [self.models[i] for i in indices]
        if not selected or x.shape[1] != len(selected):
            raise ValueError("Data and estimator counts differ")
        modes = {m.training for m in selected}
        if len(modes) != 1:
            raise ValueError("Mixed training/evaluation modes")
        self.template.train(selected[0].training)
        pd = [dict(m.named_parameters()) for m in selected]
        bd = [dict(m.named_buffers()) for m in selected]
        params = {name: torch.stack([p[name] for p in pd]) for name in pd[0]}
        buffers = {name: torch.stack([b[name] for b in bd]) for name in bd[0]}

        def call(p, b, data):
            return functional_call(self.template, (p, b), (data,), strict=True)

        logits, invalid = torch.vmap(call, in_dims=(0, 0, 1), out_dims=(1, 0), randomness="error")(
            params, buffers, x
        )
        if invalid.any():
            raise ValueError("NaN in batched encoder or classifier output")
        # Only these buffers are updated by the supported forward path.
        if selected[0].training:
            with torch.no_grad():
                for name, module in self.template.named_modules():
                    if isinstance(module, ExecutionGhostBatchNorm):
                        for suffix in ("m", "s", "gbs"):
                            key = f"{name}.{suffix}" if name else suffix
                            for j, b in enumerate(bd):
                                b[key].copy_(buffers[key][j])
        return logits


def losses(logits, y, w):
    b, k, c = logits.shape
    ce = torch.nn.functional.cross_entropy(logits.reshape(b*k, c), y.reshape(b*k), reduction="none")
    return (ce.view(b, k) * w).mean(dim=0)


def train_step(executor, optimizers, x, y, w, chunk_size, backward=None):
    if len(optimizers) != len(executor.models) or x.shape[1] != len(executor.models):
        raise ValueError("One input column and optimizer per estimator required")
    if y.shape != x.shape[:2] or w.shape != y.shape:
        raise ValueError("Input, target and weight dimensions differ")
    backward = backward or (lambda loss: loss.backward())
    values = []
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(executor.models), chunk_size):
        ids = list(range(start, min(start + chunk_size, len(executor.models))))
        for i in ids:
            optimizers[i].zero_grad()
        logits = executor.forward(ids, x[:, ids, :])
        value = losses(logits, y[:, ids], w[:, ids])
        backward(value.sum())
        for i in ids:
            optimizers[i].step()
        values.append(value.detach())
    return torch.cat(values)


def train_independent_step(executor, optimizers, batches, chunk_size, backward=None):
    """Consume already-selected per-estimator minibatches without resampling.

    Each entry is (x_i, y_i, w_i), or None if that estimator has no update.
    Different batch lengths are grouped by shape, never truncated or padded.
    Data-loader generators/seeds remain owned by the caller for each estimator.
    """
    if len(batches) != len(executor.models) or len(optimizers) != len(batches):
        raise ValueError("One batch and optimizer entry per estimator required")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    backward = backward or (lambda loss: loss.backward())
    groups = {}
    for i, batch in enumerate(batches):
        if batch is None:
            continue
        x, y, w = batch
        if x.ndim != 2 or y.shape != (len(x),) or w.shape != (len(x),):
            raise ValueError("Invalid per-estimator minibatch")
        key = tuple((tuple(t.shape), t.dtype, t.device) for t in batch)
        groups.setdefault(key, []).append(i)
    values = [None] * len(batches)
    for members in groups.values():
        for start in range(0, len(members), chunk_size):
            ids = members[start:start+chunk_size]
            x,y,w = (torch.stack([batches[i][j] for i in ids], dim=1) for j in range(3))
            for i in ids:
                optimizers[i].zero_grad()
            value = losses(executor.forward(ids,x),y,w)
            backward(value.sum())
            for j,i in enumerate(ids):
                optimizers[i].step()
                values[i] = value[j].detach()
    return values
