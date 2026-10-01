"""Artifact integrity, row alignment, aggregation and immutable reuse checks."""
import pathlib
import sys
import tempfile
import hashlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from artifacts.training_store import TrainingStore, ArtifactConflict, mean_log_density_ratio, aggregate_log_ratios


def rejects(fn, error=ValueError):
    try:
        fn()
    except error:
        return
    raise AssertionError('Invalid input accepted')


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        store = TrainingStore(root/'artifacts')
        ds = store.put_dataset(hashlib.sha256(b'data-v1').hexdigest(), 3, {'source': 'synthetic'})
        ids = np.array([2, 0, 1], dtype=np.int64)
        split = store.put_split(ds, 'heldout', ids, {'algorithm': 'synthetic_fixture', 'version': 1, 'seed': 7})
        assert store.put_split(ds, 'heldout', ids.copy(), {'algorithm': 'synthetic_fixture', 'version': 1, 'seed': 7}) == split
        rejects(lambda: store.put_split(ds, 'heldout', ids[::-1], {'algorithm': 'synthetic_fixture', 'version': 1, 'seed': 7}), ArtifactConflict)
        assert store.verify_split(split, ids.copy())
        rejects(lambda: store.verify_split(split, ids[::-1]))
        assert not list(store.blobs.glob('*.npy')), 'Split indices were persisted'
        swapped = store.put_split(ds, 'reordered', ids[::-1], {'algorithm': 'synthetic_fixture', 'version': 1, 'seed': 7})
        weights = root/'weights.pt'
        weights.write_bytes(b'opaque model payload')
        models = [store.put_model(weights, {'member': m}, {'dtype': 'float32'}, [split]) for m in range(2)]
        assert models[0] != models[1]
        # Identical payload bytes deduplicate without conflating model identity.
        assert store.read(models[0])['payload'] == store.read(models[1])['payload']
        ensemble = store.put_ensemble(models, 'mean_density_ratio')
        a = np.array([0., 2., 1000.], dtype=np.float32)
        b = np.array([2., 0., 1001.], dtype=np.float32)
        scores = [store.put_scores(m, split, v, 'log_density_ratio') for m, v in zip(models, [a, b])]
        before_aggregation = store.storage_stats()
        actual = store.aggregate_scores(ensemble, scores)
        assert store.storage_stats() == before_aggregation
        aggregate = store.cache_aggregate_scores(ensemble, scores)
        np.testing.assert_array_equal(store.load_array(aggregate), actual)
        expected = np.logaddexp(a.astype('float64'), b.astype('float64')) - np.log(2.)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)
        assert actual.dtype == np.float32 and np.isfinite(actual).all()
        assert not np.allclose(actual[:2], ((a+b)/2)[:2])
        mean_p_id = store.put_ensemble(models, 'mean_probability')
        mean_log_id = store.put_ensemble(models, 'mean_log_density_ratio')
        assert len({ensemble, mean_p_id, mean_log_id}) == 3
        p_scores = store.aggregate_scores(mean_p_id, scores)
        log_scores = store.aggregate_scores(mean_log_id, scores)
        p = (1/(1+np.exp(-a[:2].astype('float64'))) + 1/(1+np.exp(-b[:2].astype('float64'))))/2
        np.testing.assert_allclose(p_scores[:2], np.log(p/(1-p)), rtol=1e-6)
        np.testing.assert_allclose(log_scores, (a+b)/2, rtol=1e-6)
        assert np.isfinite(p_scores).all()
        assert not np.allclose(p_scores[:2], actual[:2])
        symmetric = aggregate_log_ratios([np.array([1000.], dtype=np.float32),
                                         np.array([-1000.], dtype=np.float32)], 'mean_probability')
        np.testing.assert_array_equal(symmetric, np.zeros(1, dtype=np.float32))
        changed_order = store.put_scores(models[1], swapped, b, 'log_density_ratio')
        rejects(lambda: store.aggregate_scores(ensemble, [scores[0], changed_order]))
        rejects(lambda: store.put_scores(ensemble, split, a, 'class_logits'))
        rejects(lambda: store.put_scores(models[0], split, a.astype('float64'), 'log_density_ratio'))
        rejects(lambda: mean_log_density_ratio([]))
        rejects(lambda: store.put_ensemble([models[0], models[0]], 'mean_probability'))
        rejects(lambda: store.read('../escape'))
        before = store.storage_stats()
        assert store.cache_aggregate_scores(ensemble, scores) == aggregate
        assert store.storage_stats() == before
        store.payload_path(scores[0]).write_bytes(b'corruption')
        rejects(lambda: store.load_array(scores[0]))
    print('PASS: immutable reuse, payload deduplication, float32 aggregation, row identity and corruption checks')


if __name__ == '__main__':
    main()
