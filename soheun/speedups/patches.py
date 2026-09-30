"""Opt-in execution patches for the #4 independent training path.

Each patch leaves every number unchanged: weights, running statistics,
Adam/scheduler state, histories, best weights and predictions. bench.py
records a fingerprint so that claim is checked, not assumed (see README.md).
Install with install([...]) before the training modules build anything.

    nosync      keep per-step loss records on the GPU; copy them once per epoch.
                Collapse nan_check's one-sync-per-parameter into one sync.
    fast_gbn    GhostBatchNorm reads Python shadows of ngb/eta/eps instead of
                0-dim CUDA buffers (each `.view(self.ngb, self.gbs * ...)` was
                an `.item()` device sync). Buffers stay registered; gbs is
                written back before any state_dict, so checkpoints are unchanged.
    fast_reinforce  phase5/slice_rewrite: reshape instead of slice-and-cat.
    graphs      CUDA-graph each member's forward+backward per full batch shape.
                Needs fast_gbn. Adam, the loss, tail batches and validation stay
                eager. FvTClassifier's NaN check becomes a device flag read at
                epoch end: a NaN stops training at the epoch end, not the step.

Fixed-batch measurements on one L40 project about 2.6x full-schedule throughput
for five workers with MPS/all patches (MPS alone: 1.6x). Full-schedule artifact
and recovery validation is required before treating this as a production gain.

These are monkeypatches over the PR #7 sources. The guards below refuse to run
if the transcribed source they replace has changed.
"""
from __future__ import annotations

import torch


# --------------------------------------------------------------------- nosync

def install_nosync() -> None:
    import independent_training as it
    import stacked_fvt
    import stacked_attention_classifier

    original_epoch_losses = it.epoch_losses

    def record(module, prefix, batches, values):
        # Same tensors as the original record(), minus the per-step .cpu().
        active = next(b for b in batches if b is not None)
        zero = active[0].new_zeros(())
        losses = torch.stack([v.detach() if v is not None else zero for v in values]).view(1, -1)
        weights = torch.stack([b[2].sum() if b is not None else zero for b in batches]).detach().view(1, -1)
        counts = torch.tensor([[len(b[1]) if b is not None else 0 for b in batches]])
        pending = module.__dict__.setdefault('_pending_records', {}).setdefault(prefix, [])
        pending.append((losses, weights, counts))

    def flush(module, prefix):
        pending = module.__dict__.get('_pending_records', {}).pop(prefix, [])
        if not pending:
            return
        # One device->host copy per epoch. Row values are bit-identical to the
        # per-step copies; the CPU reduction in epoch_losses is unchanged.
        losses = torch.cat([p[0] for p in pending], 0).cpu()
        weights = torch.cat([p[1] for p in pending], 0).cpu()
        counts = torch.cat([p[2] for p in pending], 0)
        for suffix, rows in (('losses_per_stack', losses), ('weights_per_stack', weights),
                             ('batch_sizes', counts)):
            name = prefix + '_' + suffix
            previous = getattr(module, name)
            setattr(module, name, torch.cat((previous, rows), 0) if previous.numel() else rows)

    def epoch_losses(module, prefix):
        flush(module, prefix)
        return original_epoch_losses(module, prefix)

    def nan_check(self):
        params = list(self.named_parameters())
        if not params:
            return
        flags = torch.stack([torch.isnan(p).any() for _, p in params]).cpu()
        if flags.any():
            name = params[int(flags.nonzero()[0])][0]
            print("NaN found in parameter:", name)
            raise ValueError(f"NaN found in parameter: {name}")

    it.record = record
    it.epoch_losses = epoch_losses
    stacked_fvt.epoch_losses = epoch_losses  # imported by name there
    stacked_attention_classifier.epoch_losses = epoch_losses
    stacked_fvt.StackedFvTClassifier.nan_check = nan_check
    stacked_attention_classifier.StackedAttentionClassifier.nan_check = nan_check


# ------------------------------------------------------------------- fast_gbn

GBN_INPLACE = {'on': False}

def install_fast_gbn() -> None:
    import network_blocks as nb

    cls = nb.GhostBatchNorm1d
    original_forward = cls.forward
    original_save = cls._save_to_state_dict
    original_load = cls._load_from_state_dict

    def shadows(self):
        s = self.__dict__.get('_py_shadows')
        if s is None:
            # float(item()) of the float32 buffer: the exact float32 value, so
            # eta*m and (1-eta) round exactly as with the 0-dim tensor.
            s = {'ngb': int(self.ngb.item()), 'eta': float(self.eta.item()),
                 'eps': float(self.eps.item())}
            self.__dict__['_py_shadows'] = s
        return s

    def forward(self, x, mask=None, debug=False):
        if not self.training or mask is not None:
            return original_forward(self, x, mask, debug)
        s = shadows(self)
        ngb, eta, eps = s['ngb'], s['eta'], s['eps']
        batch_size = x.shape[0]
        pixels = x.shape[2]
        gbs = batch_size // ngb
        self.__dict__['_py_gbs'] = gbs
        x = x.transpose(1, 2).contiguous().view(ngb, gbs * pixels, self.features, 1)
        gbm = x.mean(dim=1, keepdim=True) if self.bias.requires_grad else 0
        gbv = x.var(dim=1, keepdim=True)
        gbs_t = (gbv + eps).sqrt()
        bm = gbm.detach().mean(dim=0) if self.bias.requires_grad else 0
        bs = gbs_t.detach().mean(dim=0)
        x = x - gbm
        x = x / gbs_t
        x = x.view(batch_size, pixels, self.features)
        x = self.gamma * x
        x = x + self.bias
        x = x.transpose(1, 2)
        if GBN_INPLACE['on']:
            # Same values written into the existing buffers, so a captured
            # CUDA graph keeps reading and writing the live running stats.
            self.m.copy_(eta * self.m + (1 - eta) * bm)
            self.s.copy_(eta * self.s + (1 - eta) * bs)
        else:
            self.m = eta * self.m + (1 - eta) * bm
            self.s = eta * self.s + (1 - eta) * bs
        return x

    def save(self, destination, prefix, keep_vars):
        gbs = self.__dict__.get('_py_gbs')
        if gbs is not None:
            # What the original forward stored: batch_size // ngb as a long
            # tensor on ngb's device.
            self._buffers['gbs'] = torch.tensor(gbs, dtype=torch.long, device=self.ngb.device)
            self.__dict__['_py_gbs'] = None
        return original_save(self, destination, prefix, keep_vars)

    def load(self, state_dict, prefix, *args, **kwargs):
        self.__dict__.pop('_py_shadows', None)
        self.__dict__['_py_gbs'] = None
        return original_load(self, state_dict, prefix, *args, **kwargs)

    cls.forward = forward
    cls._save_to_state_dict = save
    cls._load_from_state_dict = load


def check_fast_gbn_forward_matches_source() -> None:
    """Refuse to run if the original training path differs from what forward()
    above transcribes (unmasked branch)."""
    import inspect
    import network_blocks as nb
    src = inspect.getsource(nb.GhostBatchNorm1d.forward)
    for needle in ("self.gbs = batch_size // self.ngb",
                   "gbv = x.var(dim=1, keepdim=True)",
                   "gbs = (gbv + self.eps).sqrt()",
                   "bs = gbs.detach().mean(dim=0)",
                   "self.m = self.eta * self.m + (1 - self.eta) * bm",
                   "self.s = self.eta * self.s + (1 - self.eta) * bs"):
        if needle not in src:
            raise RuntimeError(f"GhostBatchNorm1d.forward changed; re-transcribe: {needle}")


# ------------------------------------------------------------- fast_reinforce

def install_fast_reinforce() -> None:
    # Portable versions of the already verified slice_rewrite functions.
    # Importing that benchmark script changes cwd to a machine-specific checkout.
    import network_blocks as nb

    def dijet(self, j, d):
        n = j.shape[0]
        d = torch.cat((j.reshape(n, self.dim_d, 6, 2), d.unsqueeze(-1)), dim=3)
        return self.conv(d.reshape(n, self.dim_d, 18))

    def quadjet(self, d, q):
        n = d.shape[0]
        q = torch.stack((self.sym(d), torch.abs(self.antisym(d)), q), dim=3)
        return self.conv(q.reshape(n, self.dim_q, 9))

    nb.DijetReinforceLayer.forward = dijet
    nb.QuadjetReinforceLayer.forward = quadjet


# --------------------------------------------------------------------- graphs

class _Wrap(torch.nn.Module):
    # make_graphed_callables replaces a module's forward, one input shape per
    # module; a thin wrapper per (member, shape) shares the member's parameters.
    def __init__(self, member):
        super().__init__()
        self.member = member

    def forward(self, x):
        return self.member(x)


def install_graphs() -> None:
    """CUDA-graph each member's forward+backward for full-size training batches.

    Adam, the loss and all partial (tail) batches stay eager. Validation stays
    eager. Needs fast_gbn (no .item() inside capture) and switches it to
    in-place running-stat updates. make_graphed_callables' warm-up passes
    update GhostBatchNorm running stats, so those buffers are restored after
    capture. NaN checks in FvTClassifier.forward become a device-side flag
    that nan_check reads at epoch end (the result is identical unless training
    would have raised).
    """
    import fvt_classifier
    import independent_training as it
    import stacked_fvt

    GBN_INPLACE['on'] = True

    def fvt_forward(self, x):
        q = self.encoder(x)
        class_score = self.attention_classifier(q)
        flag = self.__dict__.get('_nan_flag')
        if flag is None or flag.device != q.device:
            flag = torch.zeros((), dtype=torch.bool, device=q.device)
            self.__dict__['_nan_flag'] = flag
        flag.logical_or_(torch.isnan(q).any() | torch.isnan(class_score).any())
        return class_score
    fvt_classifier.FvTClassifier.forward = fvt_forward

    def nan_check(self):
        params = list(self.named_parameters())
        flags = [torch.isnan(p).any() for _, p in params]
        members = [m for m in self.modules() if '_nan_flag' in m.__dict__]
        flags += [m.__dict__['_nan_flag'] for m in members]
        if not flags:
            return
        host = torch.stack(flags).cpu()
        if host.any():
            k = int(host.nonzero()[0])
            where = params[k][0] if k < len(params) else 'forward output'
            print("NaN found in:", where)
            raise ValueError(f"NaN found in: {where}")
    stacked_fvt.StackedFvTClassifier.nan_check = nan_check

    stats = {'captures': 0, 'capture_s': 0.0, 'replays': 0, 'eager': 0}
    it.GRAPH_STATS = stats

    def capture(member, x):
        import time
        for mod in member.modules():
            if isinstance(mod, torch.nn.Dropout) and mod.p > 0:
                raise RuntimeError('active dropout: graph capture would change RNG use')
        t0 = time.perf_counter()
        snapshot = [(b, b.detach().clone()) for b in member.buffers()]
        flag = member.__dict__.get('_nan_flag')
        flag_before = flag.clone() if flag is not None else None
        torch.cuda.synchronize()
        graphed = torch.cuda.make_graphed_callables(_Wrap(member), (x.detach().clone(),), allow_unused_input=True)
        torch.cuda.synchronize()
        with torch.no_grad():
            for b, saved in snapshot:
                b.copy_(saved)
            if flag_before is not None:
                member.__dict__['_nan_flag'].copy_(flag_before)
            elif '_nan_flag' in member.__dict__:
                member.__dict__['_nan_flag'].zero_()
        stats['captures'] += 1
        stats['capture_s'] += time.perf_counter() - t0
        return graphed

    def forward_for(module, i, member, x):
        batch_size = module.datamodule.batch_size
        cache = module.__dict__.setdefault('_graph_cache', {})
        key = (i, tuple(x.shape))
        g = cache.get(key)
        if g is None:
            if x.shape[0] != batch_size:
                stats['eager'] += 1
                return member(x)
            for k in [k for k in cache if k[1][0] != batch_size]:
                del cache[k]  # an earlier milestone's shape never recurs
            g = cache[key] = capture(member, x)
        stats['replays'] += 1
        return g(x)

    original_step = it.step

    def step(module, batch, members, *, training, chunk_size=0):
        if not training or chunk_size:
            return original_step(module, batch, members, training=training, chunk_size=chunk_size)
        # the eager, chunk_size=0 training branch of independent_training.step
        batches = [tuple(t.to(module.device) for t in b) if b is not None else None for b in batch['members']]
        if len(batches) != len(members):
            raise ValueError('Wrong number of estimator minibatches')
        optimizers = module.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        values = [None] * len(members)
        for i, b in enumerate(batches):
            if b is None:
                continue
            x, y, w = b
            optimizers[i].zero_grad()
            loss = (module.ce_loss(forward_for(module, i, members[i], x), y) * w).mean()
            module.manual_backward(loss)
            optimizers[i].step()
            values[i] = loss
        it.record(module, 'train', batches, values)
        if hasattr(module.datamodule, 'note_training_batch'):
            module.datamodule.note_training_batch()

    it.step = step
    stacked_fvt.independent_step = step


def check_step_matches_source() -> None:
    import inspect
    import independent_training as it
    src = inspect.getsource(it.step)
    for needle in ("optimizers[i].zero_grad()",
                   "loss = (module.ce_loss(members[i](x), y) * w).mean()",
                   "module.manual_backward(loss)", "optimizers[i].step()",
                   "record(module, 'train' if training else 'val', batches, values)"):
        if needle not in src:
            raise RuntimeError(f"independent_training.step changed; re-transcribe: {needle}")


def install(names: list[str]) -> None:
    allowed = ('nosync', 'fast_gbn', 'fast_reinforce', 'graphs')
    if len(set(names)) != len(names) or any(name not in allowed for name in names):
        raise ValueError('Unknown or duplicate execution patch')
    names = [name for name in allowed if name in names]
    if 'graphs' in names and 'fast_gbn' not in names:
        raise ValueError('graphs requires fast_gbn')
    for name in names:
        if name == 'nosync':
            install_nosync()
        elif name == 'fast_gbn':
            check_fast_gbn_forward_matches_source()
            install_fast_gbn()
        elif name == 'fast_reinforce':
            install_fast_reinforce()
        elif name == 'graphs':
            check_step_matches_source()
            install_graphs()
        else:
            raise ValueError(f"unknown patch {name}")
