"""Immutable, content-checked training artifacts, separate from legacy caches.

Model payloads remain opaque bytes; this module never unpickles a model. Numeric
arrays are non-pickle NPY files. Semantic records are immutable: reusing the same
identity with different data raises instead of silently replacing prior results.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import numpy as np


class ArtifactConflict(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def mean_log_density_ratio(member_log_ratios):
    """Streaming log(arithmetic mean of member ratios), always float32.

This is neither mean probability odds nor mean logits. No exponentiation of
large individual log ratios is necessary. All members must share row ordering.
"""
    total = None
    count = 0
    for values in member_log_ratios:
        values = np.asarray(values)
        if values.dtype != np.float32 or values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError('Expected finite one-dimensional float32 member log ratios')
        if total is None:
            total = values.copy()
        else:
            if total.shape != values.shape:
                raise ValueError('Member prediction lengths differ')
            np.logaddexp(total, values, out=total)
        count += 1
    if not count:
        raise ValueError('An ensemble needs at least one member')
    return total - np.float32(np.log(count))


def aggregate_log_ratios(member_log_ratios, aggregation):
    """Return a float32 log-ratio representation of the chosen ensemble rule.

    Mean probability uses log-sigmoid sums, avoiding rounded probabilities of
    exactly zero/one when individual float32 logits are large in magnitude.
    """
    if aggregation == 'mean_density_ratio':
        return mean_log_density_ratio(member_log_ratios)
    if aggregation not in ('mean_probability', 'mean_log_density_ratio'):
        raise ValueError('Unsupported aggregation rule')
    count = 0
    average = log_p_sum = log_q_sum = None
    shape = None
    for values in member_log_ratios:
        values = np.asarray(values)
        if values.dtype != np.float32 or values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError('Expected finite one-dimensional float32 member log ratios')
        if shape is not None and values.shape != shape:
            raise ValueError('Member prediction lengths differ')
        shape = values.shape
        count += 1
        if aggregation == 'mean_log_density_ratio':
            if average is None:
                average = values.copy()
            else:
                average *= np.float32((count - 1) / count)
                average += values / np.float32(count)
        else:
            log_p = -np.logaddexp(np.float32(0), -values)
            log_q = -np.logaddexp(np.float32(0), values)
            if log_p_sum is None:
                log_p_sum, log_q_sum = log_p, log_q
            else:
                np.logaddexp(log_p_sum, log_p, out=log_p_sum)
                np.logaddexp(log_q_sum, log_q, out=log_q_sum)
    if not count:
        raise ValueError('An ensemble needs at least one member')
    return average if aggregation == 'mean_log_density_ratio' else log_p_sum - log_q_sum


AGGREGATIONS = {
    'mean_density_ratio': ('log_mean_density_ratio', 'ordered_float32_logaddexp_v1'),
    'mean_probability': ('log_odds_mean_probability', 'ordered_float32_logsigmoid_sums_v1'),
    'mean_log_density_ratio': ('mean_log_density_ratio', 'ordered_float32_running_mean_v1'),
}


class TrainingStore:
    def __init__(self, root):
        self.root = Path(root)
        self.records = self.root/'records'
        self.blobs = self.root/'blobs'
        self.records.mkdir(parents=True, exist_ok=True)
        self.blobs.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _check_id(value):
        if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
            raise ValueError('Invalid artifact ID')

    def _publish(self, path, data):
        # Link a fully fsynced unique temporary file: concurrent writers cannot
        # replace an existing artifact, and readers never see partial content.
        handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pending-', delete=False)
        tmp = Path(handle.name)
        try:
            with handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(tmp, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ArtifactConflict(f'Existing artifact differs: {path.name}')
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            tmp.unlink(missing_ok=True)

    def _record(self, kind, identity, payload=None):
        key = sha(canonical({'schema': 1, 'kind': kind, 'identity': identity}))
        record = {'schema': 1, 'kind': kind, 'identity': identity, 'id': key, 'payload': payload}
        self._publish(self.records/f'{key}.json', canonical(record))
        return key

    def _blob(self, data, suffix):
        key = sha(data)
        name = key + suffix
        self._publish(self.blobs/name, data)
        return {'sha256': key, 'filename': name, 'bytes': len(data)}

    def read(self, artifact_id, expected_kind=None):
        self._check_id(artifact_id)
        record = json.loads((self.records/f'{artifact_id}.json').read_text())
        expected_id = sha(canonical({k: record[k] for k in ('schema', 'kind', 'identity')}))
        if record['id'] != artifact_id or expected_id != artifact_id:
            raise ValueError('Artifact identity checksum failed')
        if expected_kind and record['kind'] != expected_kind:
            raise ValueError(f'Expected {expected_kind}, got {record["kind"]}')
        return record

    def payload_path(self, artifact_id):
        record = self.read(artifact_id)
        payload = record['payload']
        if not isinstance(payload, dict):
            raise ValueError('Artifact has no payload')
        self._check_id(payload['sha256'])
        if payload['filename'] not in (payload['sha256']+'.npy', payload['sha256']+'.weights'):
            raise ValueError('Invalid payload filename')
        path = self.blobs/payload['filename']
        if path.stat().st_size != payload['bytes'] or sha(path.read_bytes()) != payload['sha256']:
            raise ValueError('Artifact payload checksum failed')
        return path

    def put_array(self, kind, identity, values):
        values = np.asarray(values)
        if values.dtype.hasobject:
            raise ValueError('Object arrays are not supported')
        buffer = io.BytesIO()
        np.save(buffer, values, allow_pickle=False)
        payload = self._blob(buffer.getvalue(), '.npy')
        payload.update(dtype=values.dtype.str, shape=list(values.shape))
        return self._record(kind, identity, payload)

    def load_array(self, artifact_id, expected_kind=None):
        record = self.read(artifact_id, expected_kind)
        values = np.load(self.payload_path(artifact_id), allow_pickle=False)
        if list(values.shape) != record['payload']['shape'] or values.dtype.str != record['payload']['dtype']:
            raise ValueError('Array metadata differs from payload')
        return values

    def put_dataset(self, source_fingerprint, row_count, selection_recipe):
        self._check_id(source_fingerprint)
        if not isinstance(row_count, int) or row_count < 0:
            raise ValueError('Invalid row count')
        return self._record('dataset', {'source_fingerprint': source_fingerprint,
                            'row_count': row_count, 'selection_recipe': selection_recipe})

    def put_split(self, dataset_id, split_name, indices, recipe):
        dataset = self.read(dataset_id, 'dataset')
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype != np.int64:
            raise ValueError('Split indices must be one-dimensional int64')
        if len(indices) and (indices.min() < 0 or indices.max() >= dataset['identity']['row_count']):
            raise ValueError('Split index outside source dataset')
        if not isinstance(recipe, dict) or not {'algorithm', 'version', 'seed'} <= recipe.keys():
            raise ValueError('A split needs a versioned reconstruction algorithm and seed')
        # Store a recipe and checksum, never a full per-member index array.
        fingerprint = sha(indices.astype('<i8', copy=False).tobytes())
        return self._record('split', {'dataset_id': dataset_id, 'name': split_name, 'recipe': recipe},
                            {'storage': 'recipe_only', 'shape': [len(indices)],
                             'index_sha256': fingerprint, 'index_encoding': 'little_endian_int64'})

    def verify_split(self, split_id, reconstructed_indices):
        record = self.read(split_id, 'split')
        indices = np.asarray(reconstructed_indices)
        if indices.dtype != np.int64 or list(indices.shape) != record['payload']['shape']:
            raise ValueError('Reconstructed split shape/dtype differs')
        fingerprint = sha(indices.astype('<i8', copy=False).tobytes())
        if fingerprint != record['payload']['index_sha256']:
            raise ValueError('Reconstructed split ordering differs from recorded fingerprint')
        return True

    def put_model(self, weights_path, estimator_identity, training_recipe, split_ids):
        for key in split_ids:
            self.read(key, 'split')
        identity = {'estimator': estimator_identity, 'training_recipe': training_recipe,
                    'split_ids': list(split_ids)}
        payload = self._blob(Path(weights_path).read_bytes(), '.weights')
        return self._record('model', identity, payload)

    def put_ensemble(self, member_ids, aggregation):
        member_ids = list(member_ids)
        if not member_ids or len(set(member_ids)) != len(member_ids):
            raise ValueError('Ensemble members must be nonempty and distinct')
        if aggregation not in AGGREGATIONS:
            raise ValueError('Unsupported aggregation rule')
        for key in member_ids:
            self.read(key, 'model')
        # Preserve explicit reduction order for reproducible float32 aggregation.
        return self._record('ensemble', {'members': member_ids, 'aggregation': aggregation,
                                        'reduction': AGGREGATIONS[aggregation][1]})

    @staticmethod
    def _score_identity(owner_id, split_id, representation, inference_recipe=None):
        identity = {'owner_id': owner_id, 'split_id': split_id,
                    'representation': representation, 'row_order': 'split_index_order'}
        if inference_recipe is not None:
            identity['inference_recipe'] = inference_recipe
        return identity

    def find_scores(self, owner_id, split_id, representation, inference_recipe=None):
        identity = self._score_identity(owner_id, split_id, representation, inference_recipe)
        key = sha(canonical({'schema': 1, 'kind': 'scores', 'identity': identity}))
        if not (self.records/f'{key}.json').is_file():
            return None
        values = self.load_array(key, 'scores')
        n = self.read(split_id, 'split')['payload']['shape'][0]
        shape = (n, 2) if representation == 'class_logits' else (n,)
        if values.dtype != np.float32 or values.shape != shape or not np.isfinite(values).all():
            raise ValueError('Cached scores violate the declared layout')
        return key

    def put_scores(self, owner_id, split_id, values, representation, inference_recipe=None):
        owner = self.read(owner_id)
        if owner['kind'] not in ('model', 'ensemble'):
            raise ValueError('Scores require a model or ensemble owner')
        split = self.read(split_id, 'split')
        values = np.asarray(values)
        allowed = ('class_logits', 'log_density_ratio') if owner['kind'] == 'model' else (AGGREGATIONS[owner['identity']['aggregation']][0],)
        if representation not in allowed:
            raise ValueError('Representation does not match owner/aggregation')
        shape = (split['payload']['shape'][0], 2) if representation == 'class_logits' else (split['payload']['shape'][0],)
        if values.dtype != np.float32 or values.shape != shape or not np.isfinite(values).all():
            raise ValueError('Scores must be finite float32 with the split row count and declared layout')
        return self.put_array('scores', self._score_identity(owner_id, split_id, representation, inference_recipe), values)

    def aggregate_scores(self, ensemble_id, member_score_ids):
        """Compute derived values in memory without writing score artifacts."""
        ensemble = self.read(ensemble_id, 'ensemble')
        members = ensemble['identity']['members']
        if len(member_score_ids) != len(members):
            raise ValueError('Missing ensemble members')
        records = [self.read(key, 'scores') for key in member_score_ids]
        split_id = records[0]['identity']['split_id']
        for owner, record in zip(members, records):
            identity = record['identity']
            if identity['owner_id'] != owner or identity['split_id'] != split_id:
                raise ValueError('Member owner or row order differs')
            if identity['representation'] not in ('class_logits', 'log_density_ratio'):
                raise ValueError('Invalid member score representation')
        def log_ratios():
            for key, record in zip(member_score_ids, records):
                values = self.load_array(key, 'scores')
                yield values[:, 1] - values[:, 0] if record['identity']['representation'] == 'class_logits' else values
        rule = ensemble['identity']['aggregation']
        values = aggregate_log_ratios(log_ratios(), rule)
        return values

    def cache_aggregate_scores(self, ensemble_id, member_score_ids):
        """Explicit optional analysis cache; raw member scores remain authoritative."""
        values = self.aggregate_scores(ensemble_id, member_score_ids)
        rule = self.read(ensemble_id, 'ensemble')['identity']['aggregation']
        split_id = self.read(member_score_ids[0], 'scores')['identity']['split_id']
        return self.put_scores(ensemble_id, split_id, values, AGGREGATIONS[rule][0],
                               {'member_score_ids': list(member_score_ids), 'aggregation': rule})

    def storage_stats(self):
        records = list(self.records.glob('*.json'))
        blobs = list(self.blobs.iterdir())
        blobs = [p for p in blobs if p.is_file() and not p.name.startswith('.pending-')]
        return {'records': len(records), 'unique_payloads': len(blobs),
                'record_bytes': sum(p.stat().st_size for p in records),
                'payload_bytes': sum(p.stat().st_size for p in blobs)}
