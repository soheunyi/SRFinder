"""Export one member's unaggregated float32 log ratios with checked row ordering.

The caller supplies a factory matching the recorded model recipe and batches
of (source row indices, features). This is independent of physical file layout.
"""
import numpy as np


def export_member_scores(store, model_id, split_id, reconstructed_indices,
                         feature_batches, model_factory, device='cpu', *, reuse_existing=True, inference_recipe=None):
    import torch
    record = store.read(model_id, 'model')
    indices = np.asarray(reconstructed_indices)
    store.verify_split(split_id, indices)
    weights_path = store.payload_path(model_id)
    if inference_recipe is None:
        inference_recipe = {'version': 1, 'mode': 'eval', 'dtype': 'float32',
                            'device_type': torch.device(device).type,
                            'torch_version': str(torch.__version__),
                            'matmul_precision': torch.get_float32_matmul_precision(),
                            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}
    if reuse_existing:
        existing = store.find_scores(model_id, split_id, 'log_density_ratio', inference_recipe)
        if existing is not None:
            return existing
    model = model_factory(record)
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for value in list(model.parameters()) + list(model.buffers()):
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError('Model must preserve the float32 training recipe')
    values = np.empty(len(indices), dtype=np.float32)
    cursor = 0
    with torch.inference_mode():
        for row_indices, features in feature_batches:
            row_indices = np.asarray(row_indices)
            if row_indices.dtype != np.int64 or row_indices.ndim != 1:
                raise ValueError('Batch row indices must be one-dimensional int64')
            end = cursor + len(row_indices)
            if end > len(indices) or not np.array_equal(row_indices, indices[cursor:end]):
                raise ValueError('Prediction batch is not in the declared evaluation-event order')
            if features.dtype != torch.float32 or len(features) != len(row_indices):
                raise ValueError('Features must be aligned float32 rows')
            if len(row_indices) == 0:
                continue
            logits = model(features.to(device))
            if logits.dtype != torch.float32 or tuple(logits.shape) != (len(row_indices), 2):
                raise ValueError('Expected two-class float32 logits')
            log_ratio = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            if not np.isfinite(log_ratio).all():
                raise ValueError('Nonfinite log-density-ratio prediction')
            values[cursor:end] = log_ratio
            cursor = end
    if cursor != len(indices):
        raise ValueError('Prediction batches did not cover all evaluation events')
    # Publish only after complete validated coverage. No clipping, probability
    # conversion, aggregation or split-index array is persisted here.
    return store.put_scores(model_id, split_id, values, 'log_density_ratio', inference_recipe)
