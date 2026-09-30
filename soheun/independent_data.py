"""Per-estimator epoch streams: no truncation, padding, or shared shuffle RNG."""
import hashlib
import json
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, TensorDataset


def stream_identity(hparams):
    fields = {key: hparams.get(key) for key in
              ('dataset', 'data_seed', 'train_seed', 'model_seed', 'encoder_hash', 'signal_region')}
    return hashlib.blake2b(json.dumps(fields, sort_keys=True, default=str).encode(), digest_size=16).hexdigest()


def epoch_seed(seed, epoch):
    payload = f'shuffle\x00{int(seed)}\x00{int(epoch)}'.encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'big') % (2**31)


class IndependentEpoch(Dataset):
    def __init__(self, datasets, batch_size, seeds, epoch, shuffle):
        self.datasets = datasets
        self.batch_size = batch_size
        self.orders = []
        self.ranges = []
        for dataset, seed in zip(datasets, seeds):
            ranges = [(start, min(start + batch_size, len(dataset))) for start in range(0, len(dataset), batch_size)]
            if shuffle:
                if batch_size % 32 or batch_size < 64 or len(dataset) % 32 or len(dataset) < 64:
                    raise ValueError('Training needs 32-aligned datasets and batch sizes, with at least 64 rows')
                if len(ranges) > 1 and ranges[-1][1] - ranges[-1][0] == 32:
                    ranges[-2:] = [(ranges[-2][0], ranges[-1][1])]
            self.ranges.append(ranges)
            generator = torch.Generator().manual_seed(epoch_seed(seed, epoch))
            order = torch.randperm(len(dataset), generator=generator) if shuffle else torch.arange(len(dataset))
            device = dataset.device if hasattr(dataset, 'gather') else dataset.tensors[0].device
            self.orders.append(order.to(device))

    def __len__(self):
        return max(map(len, self.ranges))

    def __getitem__(self, step):
        members = []
        for dataset, order, ranges in zip(self.datasets, self.orders, self.ranges):
            if step >= len(ranges):
                members.append(None)
                continue
            start, end = ranges[step]
            indices = order[start:end]
            members.append(dataset.gather(indices) if hasattr(dataset, 'gather')
                           else tuple(t.index_select(0, indices) for t in dataset.tensors))
        return {'members': members}


class IndependentStackedDataModule(pl.LightningDataModule):
    def __init__(self, train_datasets, val_datasets, batch_size, *, shuffle_seeds,
                 estimator_ids=None, batch_size_milestones=None, batch_size_multiplier=2,
                 num_workers=0, pin_memory=False, persistent_workers=False, storage_device=None):
        super().__init__()
        if not train_datasets or len(train_datasets) != len(val_datasets) or len(shuffle_seeds) != len(train_datasets):
            raise ValueError('One train set, validation set and shuffle seed per estimator required')
        if batch_size < 1 or batch_size_multiplier < 1:
            raise ValueError('Invalid batch-size schedule')
        if any(len(d) == 0 for d in [*train_datasets, *val_datasets]):
            raise ValueError('Every estimator needs nonempty training and validation data')
        if storage_device is not None:
            storage_device = torch.device(storage_device)
            if storage_device.type == 'cuda' and (num_workers or pin_memory):
                raise ValueError('GPU-resident data requires num_workers=0 and pin_memory=False')
            train_datasets = [TensorDataset(*(t.to(storage_device) for t in d.tensors)) for d in train_datasets]
            val_datasets = [TensorDataset(*(t.to(storage_device) for t in d.tensors)) for d in val_datasets]
        self.train_datasets = list(train_datasets)
        self.val_datasets = list(val_datasets)
        self.shuffle_seeds = list(map(int, shuffle_seeds))
        self.estimator_ids = list(estimator_ids) if estimator_ids is not None else None
        if self.estimator_ids is not None and len(self.estimator_ids) != len(train_datasets):
            raise ValueError('Estimator identity count differs')
        self.base_batch_size = int(batch_size)
        self.batch_size = int(batch_size)
        self.batch_size_milestones = sorted(set(batch_size_milestones or []))
        self.batch_size_multiplier = int(batch_size_multiplier)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self._stream_epoch = None
        self._completed_steps = 0
        self._expected_steps = 0

    def loader_for_epoch(self, epoch, training=True):
        self.batch_size = self.base_batch_size * self.batch_size_multiplier ** sum(m <= epoch for m in self.batch_size_milestones)
        data = IndependentEpoch(self.train_datasets if training else self.val_datasets,
                                self.batch_size, self.shuffle_seeds, epoch, training)
        if training and epoch != self._stream_epoch:
            self._stream_epoch = epoch
            self._completed_steps = 0
        if training:
            self._expected_steps = len(data)
        # A separate generator prevents iterator/worker creation consuming model RNG.
        return DataLoader(data, batch_size=None, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers and self.num_workers > 0,
                          generator=torch.Generator().manual_seed(epoch_seed(0, epoch)))

    def train_dataloader(self):
        return self.loader_for_epoch(self.trainer.current_epoch, True)

    def val_dataloader(self):
        return self.loader_for_epoch(self.trainer.current_epoch, False)

    def note_training_batch(self):
        self._completed_steps += 1

    def state_dict(self):
        if 0 < self._completed_steps < self._expected_steps:
            raise RuntimeError('Independent streams support epoch-boundary checkpoints only; use the last completed-epoch checkpoint')
        return {'version': 1, 'base_batch_size': self.base_batch_size,
                'batch_size_milestones': self.batch_size_milestones,
                'batch_size_multiplier': self.batch_size_multiplier,
                'shuffle_seeds': self.shuffle_seeds, 'estimator_ids': self.estimator_ids,
                'train_lengths': [len(d) for d in self.train_datasets],
                'val_lengths': [len(d) for d in self.val_datasets]}

    def load_state_dict(self, state_dict):
        # Epoch-boundary resume regenerates each permutation from seed and epoch.
        # Never silently resume a different ordering of estimators or data lengths.
        if state_dict != self.state_dict():
            raise ValueError('Independent data stream configuration differs from checkpoint')

    def validation_probe(self, limit):
        """Explicit common-prefix diagnostic only; training/evaluation loaders use all rows."""
        n = min(limit, *(len(d) for d in self.val_datasets))
        members = [d.gather(torch.arange(n, device=d.device)) if hasattr(d, 'gather')
                   else tuple(t[:n] for t in d.tensors) for d in self.val_datasets]
        return TensorDataset(*(torch.stack([values[j] for values in members], 1).cpu()
                               for j in range(len(members[0]))))


def row_counts(dm, training):
    name = 'train_datasets' if training else 'val_datasets'
    if hasattr(dm, name):
        return [len(d) for d in getattr(dm, name)]
    dataset = dm.stacked_train_dataset if training else dm.stacked_val_dataset
    return [len(dataset)] * dataset.tensors[0].shape[1]


def validation_probe(dm, limit):
    if isinstance(dm, IndependentStackedDataModule):
        return dm.validation_probe(limit)
    return TensorDataset(*(t[:limit] for t in dm.stacked_val_dataset.tensors))
