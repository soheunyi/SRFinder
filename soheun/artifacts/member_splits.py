"""Reconstruct native member row order without saving per-member index arrays.

Rows index the verified mother selection, whose pool ordering is registered in
its dataset record. Training shuffle order is separately owned by the loader.
"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from .training_store import canonical


def reconstruct_member_splits(context, *, training_alignment=32):
    if type(training_alignment) is not int or training_alignment < 0:
        raise ValueError('Training alignment must be a nonnegative integer')
    # Legacy aux_info can contain score arrays and training history. Neither
    # defines this explicit alignment/full-validation split policy.
    hp = {key: value for key, value in context.hparams.items()
          if not key.startswith('aux_info')}
    stage = hp.get('step')
    if stage not in (1, 2, 3):
        raise ValueError('Unsupported training stage')
    if not 0 < context.val_ratio < 1:
        raise ValueError('Validation fraction must be between zero and one')
    mask = np.asarray(context.ms_idx)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError('Expected a boolean selection over the verified mother rows')
    rows = np.flatnonzero(mask).astype(np.int64)
    seed = int(context.data_seed)
    if stage == 2:
        # The smeared path shuffles before encoding, then sklearn splits again.
        shuffled = pd.Series(rows).sample(frac=1, random_state=seed).to_numpy()
        train, validation = train_test_split(shuffled, test_size=context.val_ratio,
                                            random_state=seed)
    else:
        # SCDatasetInfo stores membership masks: its selection loses permutation
        # order and fetch_data restores pool/row order before pandas shuffles.
        order = np.random.RandomState(seed).permutation(len(rows))
        cut = int((1 - context.val_ratio) * len(rows))
        train, validation = [
            pd.Series(np.sort(rows[idx])).sample(frac=1, random_state=seed).to_numpy()
            for idx in (order[:cut], order[cut:])
        ]
    if training_alignment:
        train = train[:len(train) // training_alignment * training_alignment]
    root = Path(__file__).resolve().parents[1]
    recipe = {
        'algorithm': 'native_member_rows', 'version': 1, 'seed': seed,
        'stage': stage, 'context_identity': context.hash,
        'hparams_sha256': hashlib.sha256(canonical(hp)).hexdigest(),
        'selection_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
        'val_ratio': context.val_ratio, 'training_alignment': training_alignment,
        'retain_validation': True,
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__,
                     'sklearn': sklearn.__version__},
        'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ('training_info.py', 'dataset.py',
                                       'artifacts/member_splits.py')},
    }
    return (np.asarray(train, dtype=np.int64),
            np.asarray(validation, dtype=np.int64)), recipe


def publish_member_splits(store, context, *, training_alignment=32):
    rows, recipe = reconstruct_member_splits(context, training_alignment=training_alignment)
    dataset_id = context.hparams['source_dataset_id']
    if store.read(dataset_id, 'dataset')['identity']['row_count'] != len(context.ms_idx):
        raise ValueError('Member selection length differs from registered dataset')
    return tuple(store.put_split(dataset_id, f"step{recipe['stage']}_{name}", indices, recipe)
                 for name, indices in zip(('train', 'validation'), rows))


def verify_member_splits(store, context, split_ids, *, training_alignment=32):
    """Reject altered recipes, source versions, membership, or row ordering."""
    if len(split_ids) != 2:
        raise ValueError('Expected train and validation split records')
    rows, recipe = reconstruct_member_splits(context, training_alignment=training_alignment)
    for name, indices, key in zip(('train', 'validation'), rows, split_ids):
        identity = store.read(key, 'split')['identity']
        expected = {'dataset_id': context.hparams['source_dataset_id'],
                    'name': f"step{recipe['stage']}_{name}", 'recipe': recipe}
        if identity != expected:
            raise ValueError('Member split reconstruction recipe differs')
        store.verify_split(key, indices)
    return rows
