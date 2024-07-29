# returns whether a given event is in the signal region

from typing import Iterable
import numpy as np
from events_data import EventsData
import umap
import multiprocessing as mp


def np_put(p):
    n = p.size
    s = np.zeros(n, dtype=np.int32)
    i = np.arange(n, dtype=np.int32)
    np.put(s, p, i)  # s[p[i]] = i
    return s


def fast_umap(reducer: umap.UMAP, n_workers: int = 4):
    def fast_umap_inner(data: np.ndarray):
        with mp.Pool(n_workers) as pool:
            return np.vstack(
                pool.map(reducer.transform, np.array_split(data, n_workers))
            )

    return fast_umap_inner


def sort_samples_by_bins(
    embeddings: np.ndarray,
    bins: list[tuple[float, float]],  # modify get_bin_idx to use other types of bins
    seed: int = None,
):
    """
    weights: weights of the events
    bins: bins for the embedding
    bins_order: order of the bins

    1. Samples in the previous bin are always ranked higher than the samples in the next bin.
    2. Randomly rank samples in the same bin and take the first ones until the total weight exceeds the cuts on weights.
    """

    if seed is not None:
        np.random.seed(seed)

    def get_bin_idx(embeddings):
        bin_indices = np.nan * np.ones(len(embeddings))

        for i in range(len(bins)):
            current_bin = bins[i]
            bin_indices[
                (embeddings >= current_bin[0]) & (embeddings <= current_bin[1])
            ] = i

        if np.any(np.isnan(bin_indices)):
            raise ValueError("Some embeddings are not in the bins")
        return bin_indices

    bin_indices = get_bin_idx(embeddings)
    bin_indices = bin_indices + np.random.rand(
        len(embeddings)
    )  # for tie-breaking within the same bin
    sorted_idx = np.argsort(bin_indices)

    return sorted_idx


def estimate_probs_4b(
    events_data: EventsData,
    cluster_embed: np.ndarray,
    est_probs_4b: np.ndarray,
    bins: np.ndarray,
):
    assert len(est_probs_4b) == len(
        events_data
    ), "Need to provide est_probs_4b of all events"

    hist_all, _ = np.histogram(cluster_embed, bins=bins, weights=events_data.weights)

    hist_est_probs_4b, _ = np.histogram(
        cluster_embed,
        bins=bins,
        weights=events_data.weights * est_probs_4b,
    )

    return hist_est_probs_4b / hist_all


def get_regions(
    events_data: EventsData,
    reducer: umap.UMAP,
    target_idx: np.ndarray,
    w_cuts: Iterable[float],
    init_bins: int = 2000,
    max_bins: int = 100,
    update_npd: bool = True,
    p4b_method="simple",
    seed: int = None,
    reuse_embed: bool = False,
):
    """
    max_bins: maximum number of bins. In particular, min_weights_per_bin = total_target_weight / max_bins.
    target_idx: indices of the target events where we estimate ratio of 4b to 3b events.
    w_sr: fraction of signal region (in terms of the total weight)
    w_cr: fraction of control region (in terms of the total weight)
    p4b_method: method to estimate the ratio of 4b to 3b events.
    """
    assert all(0 < w_cut <= 1 for w_cut in w_cuts), "0 < w_cut <= 1"
    assert len(target_idx) == len(
        events_data
    ), "Need to provide target_idx for all events"

    if reuse_embed:
        assert "cluster_embed" in events_data.npd, "Need to set cluster_embed in npd"
        cluster_embed = events_data.npd["cluster_embed"]
    else:
        fast_reducer = fast_umap(reducer, n_workers=4)
        cluster_embed = fast_reducer(events_data.att_q_repr)

    if update_npd:
        events_data.update_npd("cluster_embed", cluster_embed)

    weights = events_data.weights

    total_target_weight = events_data[target_idx].total_weight
    min_weights_bin = total_target_weight / max_bins

    embed_bins = np.linspace(cluster_embed.min(), cluster_embed.max(), init_bins + 1)
    hist_target, _ = np.histogram(
        cluster_embed[target_idx, 0],
        bins=embed_bins,
        weights=weights[target_idx],
    )

    w_csum = 0
    merged_x_bins = [embed_bins[0]]
    for i in range(len(hist_target)):
        w_csum += hist_target[i]
        if w_csum > min_weights_bin:
            merged_x_bins.append(embed_bins[i + 1])
            w_csum = 0

    merged_x_bins[-1] = embed_bins[-1]
    merged_x_bins = np.array(merged_x_bins)

    if p4b_method == "simple":
        ratio_4b_target = estimate_probs_4b(
            events_data[target_idx],
            cluster_embed[target_idx, 0],
            events_data[target_idx].is_4b,
            bins=merged_x_bins,
        )
    elif p4b_method == "fvt":
        ratio_4b_target = estimate_probs_4b(
            events_data[target_idx],
            cluster_embed[target_idx, 0],
            events_data[target_idx].fvt_score,
            bins=merged_x_bins,
        )
    else:
        raise ValueError("Unknown p4b_method")

    # sort by ratio_4b_target and plot csum

    sorted_bin_idx = np.argsort(ratio_4b_target)[::-1]
    sorted_bins = [(merged_x_bins[i], merged_x_bins[i + 1]) for i in sorted_bin_idx]
    sample_idx_sorted = sort_samples_by_bins(
        cluster_embed[:, 0], sorted_bins, seed=seed
    )
    csum_all = np.cumsum(weights[sample_idx_sorted])
    csum_all = csum_all / csum_all[-1]
    inv_sample_idx_sorted = np.argsort(sample_idx_sorted)

    is_in_region_list = [(csum_all <= w_cut)[inv_sample_idx_sorted] for w_cut in w_cuts]

    return is_in_region_list


def get_fvt_cut_regions(
    events_data: EventsData,
    fvt_cut: float,
    w_cuts: Iterable[float],
    init_bins: int = 1000,
    max_bins: int = 50,
    seed: int = None,
    reducer_args: dict = {},
    return_reducer: bool = False,
    p4b_method="simple",
):
    """
    fvt_cut: cut on the fvt_score
    w_sr: fraction of signal region (in terms of the total weight)
    w_cr: fraction of control region (in terms of the total weight)
    """

    assert all(0 < w_cut <= 1 for w_cut in w_cuts), "0 < w_cut <= 1"
    assert 0 <= fvt_cut < 1, "0 <= fvt_cut < 1"
    assert events_data.fvt_score is not None, "Need to set fvt_score in events_data"

    if seed is not None:
        np.random.seed(seed)

    reducer = umap.UMAP(n_components=1, **reducer_args)
    fvt_exceeded = events_data.fvt_score > fvt_cut
    is_4b = events_data.is_4b

    # clusters at most 10000 points for performance reasons
    # sample it via poisson sampling

    cluster_candidates = fvt_exceeded & is_4b
    cluster_idx = np.random.choice(
        np.where(cluster_candidates)[0],
        min(10000, np.sum(cluster_candidates)),
        replace=False,
        p=events_data.weights[cluster_candidates]
        / np.sum(events_data.weights[cluster_candidates]),
    )
    cluster_events = events_data[cluster_idx]
    reducer.fit(cluster_events.att_q_repr)

    if return_reducer:
        return (
            get_regions(
                events_data,
                reducer,
                fvt_exceeded,
                w_cuts,
                init_bins=init_bins,
                max_bins=max_bins,
                seed=seed,
                p4b_method=p4b_method,
            ),
            reducer,
        )
    else:
        return get_regions(
            events_data,
            reducer,
            fvt_exceeded,
            w_cuts,
            init_bins=init_bins,
            max_bins=max_bins,
            seed=seed,
            p4b_method=p4b_method,
        )


def get_regions_via_histogram(
    events_data: EventsData,
    w_cuts: Iterable[float],
    binning_method="uniform",
    n_bins=10,
):
    """
    binning_method: uniform, quantile
    """
    assert all(0 <= w_cut <= 1 for w_cut in w_cuts), "0 <= w_cut <= 1"

    weights = events_data.weights
    total_weight = events_data.total_weight
    att_q_repr = events_data.att_q_repr
    is_3b = events_data.is_3b
    is_4b = events_data.is_4b

    if binning_method == "uniform":
        bins = [
            np.linspace(att_q_repr[:, i].min(), att_q_repr[:, i].max(), n_bins + 1)
            for i in range(att_q_repr.shape[1])
        ]
    elif binning_method == "quantile":
        bins = [
            np.quantile(att_q_repr[:, i], np.linspace(0, 1, n_bins + 1))
            for i in range(att_q_repr.shape[1])
        ]
    else:
        raise ValueError("Unknown binning_method")

    # for each att_q_repr, calculate membership of the bins

    att_q_repr_bin_idx = np.stack(
        [
            np.digitize(att_q_repr[:, i], bins[i]) - 1
            for i in range(att_q_repr.shape[1])
        ],
        axis=1,
    )
    att_q_repr_bin_idx = np.clip(att_q_repr_bin_idx, 0, n_bins - 1)

    # hist_3b, _ = np.histogramdd(att_q_repr[is_3b], bins=bins, weights=weights[is_3b])
    hist_4b, _ = np.histogramdd(att_q_repr[is_4b], bins=bins, weights=weights[is_4b])
    hist_all, _ = np.histogramdd(att_q_repr, bins=bins, weights=weights)

    is_nonzero = hist_all > 0

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_4b = hist_4b / hist_all

    sorted_idx = np.argsort(ratio_4b[is_nonzero])[::-1]

    is_in_region_list = []
    nonzero_bins = np.transpose(np.nonzero(is_nonzero))

    events_membership = -np.ones(len(events_data), dtype=int)
    designated = np.zeros(len(events_data), dtype=bool)

    for i in range(len(nonzero_bins)):
        in_ith_bin = np.all(att_q_repr_bin_idx[~designated] == nonzero_bins[i], axis=1)
        desig_idx = np.nonzero(~designated)[0][in_ith_bin]
        events_membership[desig_idx] = i
        designated[desig_idx] = True

    assert np.all(events_membership >= 0) and np.all(
        events_membership < len(nonzero_bins)
    ), "Some events are not in the bins"

    csum_all = np.cumsum(hist_all[is_nonzero][sorted_idx])
    csum_all = csum_all / csum_all[-1]

    for w_cut in w_cuts:
        is_region_bins = csum_all <= w_cut
        is_region_bins = is_region_bins[np_put(sorted_idx)]
        is_in_region_list.append(is_region_bins[events_membership])

    return is_in_region_list


def get_regions_via_probs_4b(
    events_data: EventsData,
    w_cuts: Iterable[float],
    probs_4b: np.ndarray,
):
    assert len(probs_4b) == len(events_data), "Need to provide probs_4b for all events"
    assert all(0 <= w_cut <= 1 for w_cut in w_cuts), "0 <= w_cut <= 1"

    sorted_idx = np.argsort(probs_4b)[::-1]
    csum_all = np.cumsum(events_data.weights[sorted_idx])
    csum_all = csum_all / csum_all[-1]
    inv_sorted_idx = np_put(sorted_idx)

    is_in_region_list = [(csum_all <= w_cut)[inv_sorted_idx] for w_cut in w_cuts]

    return is_in_region_list
