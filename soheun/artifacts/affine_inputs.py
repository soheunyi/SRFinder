"""In-memory bridge from aligned X2 scores to the existing affine bootstrap.

Raw member caches stay float32; the bootstrap's numerical calculations use
float64 as in the existing reference implementation. No arrays are persisted.
"""
import hashlib
import json
import numpy as np


def _weight_diagnostics(weights, signed=False):
    total=float(np.sum(weights,dtype=np.float64))
    if not np.isfinite(total) or total<=0:
        raise ValueError('Effective weights need positive finite total')
    normalized=weights/total
    squares=np.square(normalized)
    square_total=float(squares.sum())
    if not square_total>0 or not np.isfinite(square_total):
        raise ValueError('Invalid effective-weight square sum')
    return {'variance_equivalent_n':1/square_total,
            'max_absolute_normalized_weight':float(np.max(np.abs(normalized))),
            'max_squared_weight_share':float(np.max(squares)/square_total),
            'signed':bool(signed)}


def prepare_affine_inputs(event_keys, is_4b, physical_weights, log_psi, log_cr_ratio,
                          *, dataset_id, domain, lower, upper=10.):
    """Return (z3, q_scaled, z4, signed_w4), plus input/normalization diagnostics.

    event_keys are (pool ID, source row) within a pinned dataset version. The SR
    is selected with the already-frozen X1 threshold BEFORE upper clipping.
    q is scaled by one common positive factor to avoid exp(log_ratio) overflow;
    this cancels in both normalized affine endpoint weights.
    """
    if domain!='X2':
        raise ValueError('The test adapter requires the held-out X2 domain')
    if not dataset_id or not np.isfinite(lower) or not np.isfinite(upper) or lower>=upper:
        raise ValueError('Pinned dataset identity and finite L<U are required')
    keys=np.asarray(event_keys)
    labels=np.asarray(is_4b)
    weights=np.asarray(physical_weights,dtype=np.float64)
    score=np.asarray(log_psi,dtype=np.float64)
    gamma=np.asarray(log_cr_ratio,dtype=np.float64)
    n=len(keys)
    if keys.shape!=(n,2) or keys.dtype!=np.int64 or np.any(keys<0):
        raise ValueError('Event keys must be nonnegative int64 pool/row pairs')
    if any(a.shape!=(n,) for a in (labels,weights,score,gamma)) or not np.isin(labels,[0,1]).all():
        raise ValueError('All fields must be aligned vectors with binary class labels')
    if not all(np.isfinite(a).all() for a in (weights,score,gamma)):
        raise ValueError('Nonfinite input')
    order=np.lexsort((keys[:,1],keys[:,0]))
    keys=keys[order]
    if n>1 and np.any(np.all(keys[1:]==keys[:-1],axis=1)):
        raise ValueError('Duplicate event identity in one test sample')
    labels=labels[order].astype(bool)
    weights,score,gamma=(a[order] for a in (weights,score,gamma))
    if np.any(weights[~labels]<0):
        raise ValueError('3b physical weights must be nonnegative')
    in_sr=score>=lower
    selected=keys[in_sr]
    labels,weights,raw,gamma=(a[in_sr] for a in (labels,weights,score,gamma))
    if not np.any(labels) or not np.any(~labels):
        raise ValueError('Both classes need SR observations')
    z=np.minimum(raw,upper)
    z3,z4=z[~labels],z[labels]
    w3,w4=weights[~labels],weights[labels]
    logq=np.full(w3.shape,-np.inf)
    positive=w3>0
    if not positive.any():
        raise ValueError('3b total weight is zero')
    logq[positive]=gamma[~labels][positive]+np.log(w3[positive])
    shift=float(np.max(logq))
    q=np.exp(logq-shift)
    position=(z3-lower)/(upper-lower)
    plus=q*position
    minus=q*(1-position)
    plus_total,minus_total=float(plus.sum()),float(minus.sum())
    total4=float(w4.sum())
    if not plus_total>0 or not minus_total>0 or not np.isfinite(total4) or not total4>0:
        raise ValueError('Both 3b endpoint totals and signed 4b total must be positive')
    h=hashlib.sha256(json.dumps({'dataset_id':dataset_id,'domain':domain,'L':float(lower),'U':float(upper)},sort_keys=True).encode())
    for a in (selected,labels,weights,raw,gamma):
        h.update(str(a.dtype).encode());h.update(np.ascontiguousarray(a).tobytes())
    diagnostic={
        'dataset_id':dataset_id,'domain':domain,'input_sha256':h.hexdigest(),
        'lower':float(lower),'upper':float(upper),'region_selection':'raw_log_psi >= lower, before clipping',
        'event_order':'canonical pool/row; bootstrap then stable-sorts within class by score',
        'n3':len(z3),'n4':len(z4),
        'clipped_count_3b':int(np.count_nonzero(raw[~labels]>upper)),
        'clipped_count_4b':int(np.count_nonzero(raw[labels]>upper)),
        'clipped_event_fraction':float(np.mean(raw>upper)),
        'signed_total_4b':total4,'absolute_total_4b':float(np.abs(w4).sum()),
        'negative_4b_count':int(np.count_nonzero(w4<0)),
        'positive_normalizers':True,'log_q_common_scale':shift,
        'log_q_plus_normalizer':shift+np.log(upper-lower)+np.log(plus_total),
        'log_q_minus_normalizer':shift+np.log(upper-lower)+np.log(minus_total),
        'q_underflow_count':int(np.count_nonzero(positive & (q==0))),
        'weights':{'q':_weight_diagnostics(q),'q_plus':_weight_diagnostics(plus),
                   'q_minus':_weight_diagnostics(minus),'r':_weight_diagnostics(w4,signed=np.any(w4<0))},
    }
    return (z3,q,z4,w4),diagnostic
