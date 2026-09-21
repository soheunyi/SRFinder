#!/usr/bin/env python3
"""Scan seeds for the t-SNE figure on the current ensemble experiment.

Replaces the stale CR_fvt_training_v2 / seed=1 configuration behind
figures/tsne_original_repr.pdf. For each seed we embed the CR events twice --
once from the raw inputs, once from the base FvT encoder's representations --
and score how concentrated the signal is in each embedding, so the seed that
best illustrates the manuscript's claim can be chosen on evidence.
"""
import argparse
import os
import sys
import time

REPO = "/home/export/soheuny/SRFinder/soheun"
sys.path.insert(0, REPO)
os.chdir(REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde

from constants import FEATURES
from dataset import MotherSamples
from events_data import events_from_scdinfo
from signal_region import get_SR_CR_cut
from training_info import TrainingInfo

import figure_cache

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{newtxtext}\usepackage{newtxmath}"
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 15

EXPERIMENT = "smeared_fvt_training_ensemble"
SIGNAL_FILENAME = "HH4b_picoAOD.h5"
N_3B = 100_0000
NOISE_SCALE = 2.0
SIGNAL_RATIO = 0.02
SRCR = {"4b_in_SR": 0.05, "4b_in_CR": 0.95}
OUT_DIR = os.path.join(REPO, "data/tsne_seed_scan")


def knn_signal_lift(points, is_signal, weights=None, k=10):
    """How much more likely a signal event's neighbour is to be signal than a
    point drawn at random. 1.0 = no concentration.

    weights=None gives the plain point-count version; passing the event weights
    gives the weighted analogue, which is what the physics actually measures.
    """
    if is_signal.sum() < 3:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(points)
    _, idx = nn.kneighbors(points[is_signal])
    neigh = idx[:, 1:]                                   # drop self
    if weights is None:
        base = is_signal.mean()
        return float(is_signal[neigh].mean() / base) if base > 0 else float("nan")
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    base = (w * is_signal).sum() / total if total != 0 else 0.0
    if base <= 0:
        return float("nan")
    wn = w[neigh]
    denom = wn.sum(axis=1)
    good = denom != 0
    purity = (wn * is_signal[neigh]).sum(axis=1)[good] / denom[good]
    wi = w[is_signal][good]
    if wi.sum() == 0:
        return float("nan")
    return float((purity * wi).sum() / wi.sum() / base)


def dispersion_ratio(embedding, is_signal, rng, m=2000):
    """Median pairwise distance among signal points over that among all points."""
    def med(points):
        take = points[rng.choice(len(points), size=min(m, len(points)), replace=False)]
        d = np.linalg.norm(take[:, None, :] - take[None, :, :], axis=-1)
        return float(np.median(d[np.triu_indices(len(take), k=1)]))
    sig = embedding[is_signal]
    if len(sig) < 3:
        return float("nan")
    return med(sig) / med(embedding)


def load_seed(seed):
    hashes = TrainingInfo.find(
        {
            "experiment_name": EXPERIMENT,
            "dataset": lambda x: (
                x["signal_ratio"] == SIGNAL_RATIO
                and x["seed"] == seed
                and x["n_3b"] == N_3B
            ),
            "smearing": lambda x: x["noise_scale"] == NOISE_SCALE,
        },
        use_cached_metadata=True,
    )
    if len(hashes) != 15:
        raise RuntimeError(f"seed {seed}: expected 15 ensemble members, got {len(hashes)}")

    smeared = [TrainingInfo.load(h) for h in hashes]
    bases = [TrainingInfo.load(t.hparams["encoder_hash"]) for t in smeared]

    # psi = base logit - smeared logit, maximum over the ensemble (manuscript choice)
    stats_tst = np.max(
        [b.aux_info["base_fvt_logit_tst"] - s.aux_info["smeared_fvt_logit_tst"]
         for b, s in zip(bases, smeared)], axis=0)
    stats_train = np.max(
        [b.aux_info["base_fvt_logit_train"] - s.aux_info["smeared_fvt_logit_train"]
         for b, s in zip(bases, smeared)], axis=0)

    msamples = MotherSamples.load(bases[0].ms_hash)
    ms_idx = bases[0].ms_idx
    events_train = events_from_scdinfo(msamples.scdinfo[ms_idx], FEATURES, SIGNAL_FILENAME)
    events_tst = events_from_scdinfo(msamples.scdinfo[~ms_idx], FEATURES, SIGNAL_FILENAME)

    SR_cut, CR_cut = get_SR_CR_cut(stats_train, events_train, SRCR)
    CR_idx = (stats_tst >= CR_cut) & (stats_tst < SR_cut)
    return events_tst[CR_idx], bases


def embed_seed(seed, n_points, tsne_kwargs, device, n_members=1):
    """Embeddings + class masks, cached: t-SNE costs ~90 s/seed, styling is free."""
    cache_key = figure_cache.key(
        "tsne_seed_scan.embedding_v2",
        EXPERIMENT, SIGNAL_RATIO, N_3B, NOISE_SCALE, SRCR, seed, n_points, tsne_kwargs,
        n_members,
    )
    cached = figure_cache.load(cache_key)
    if cached is not None:
        return cached

    events_CR, base_tinfos = load_seed(seed)
    rng = np.random.default_rng(seed)
    take = rng.choice(len(events_CR), size=min(n_points, len(events_CR)), replace=False)
    events = events_CR[take]

    # Every ensemble member has its own encoder but the same train/test split, so
    # their representations of the same events can be concatenated. Each member's
    # block is standardised first, otherwise one member's scale dominates t-SNE.
    blocks = []
    for tinfo in base_tinfos[:n_members]:
        model = tinfo.load_trained_model("best")
        model.eval()
        model.to(device)
        with torch.no_grad():
            q_repr, _ = model.representations(events.X_torch)
        block = q_repr.reshape(len(q_repr), -1).cpu().numpy().astype(np.float64)
        block -= block.mean(axis=0)
        scale = block.std(axis=0)
        block /= np.where(scale > 0, scale, 1.0)
        blocks.append(block / np.sqrt(len(base_tinfos[:n_members])))
    repr_flat = np.concatenate(blocks, axis=1)
    raw_flat = events.X_torch.reshape(len(events.X_torch), -1).cpu().numpy()

    emb_orig = TSNE(**tsne_kwargs).fit_transform(raw_flat)
    emb_repr = TSNE(**tsne_kwargs).fit_transform(repr_flat)
    # Measured in the FULL-dimensional spaces, not in the 2-D embedding: t-SNE
    # coordinates carry their own randomness, so ranking seeds on them confounds
    # the representation with the embedding.
    is_signal = np.asarray(events.is_signal)
    w = np.asarray(events.weights, dtype=np.float64)
    space = {
        "raw": {"lift": knn_signal_lift(raw_flat, is_signal),
                "lift_w": knn_signal_lift(raw_flat, is_signal, weights=w)},
        "repr": {"lift": knn_signal_lift(repr_flat, is_signal),
                 "lift_w": knn_signal_lift(repr_flat, is_signal, weights=w)},
    }
    payload = {
        "masks": {"is_3b": np.asarray(events.is_3b),
                  "is_bg4b": np.asarray(events.is_bg4b),
                  "is_signal": is_signal},
        "weights": w,
        "emb_orig": emb_orig,
        "emb_repr": emb_repr,
        "space": space,
    }
    return figure_cache.save(cache_key, payload)


def signal_density_contours(axis, emb, is_signal, levels=(0.5, 0.7, 0.9)):
    """Filled contours of the signal density, so concentration is visible even
    though the signal is well under 1% of the points."""
    pts = emb[is_signal]
    if len(pts) < 10:
        return
    kde = gaussian_kde(pts.T)
    pad = 0.05 * (emb.max(axis=0) - emb.min(axis=0))
    xs = np.linspace(emb[:, 0].min() - pad[0], emb[:, 0].max() + pad[0], 220)
    ys = np.linspace(emb[:, 1].min() - pad[1], emb[:, 1].max() + pad[1], 220)
    gx, gy = np.meshgrid(xs, ys)
    z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    # contour at the smallest region holding each fraction of the signal mass
    flat = np.sort(z.ravel())[::-1]
    mass = np.cumsum(flat) / flat.sum()
    cuts = sorted(flat[np.searchsorted(mass, lv)] for lv in levels)
    axis.contourf(gx, gy, z, levels=cuts + [z.max()],
                  colors=["#2ca02c"], alpha=0.16, zorder=2.5)
    axis.contour(gx, gy, z, levels=cuts, colors="#2ca02c",
                 linewidths=1.0, alpha=0.75, zorder=2.6)


def draw(seed, masks, emb_orig, emb_repr, path, raster_dpi=200, contours=False):
    is_3b, is_bg4b, is_signal = masks["is_3b"], masks["is_bg4b"], masks["is_signal"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for axis, emb, title in [
        (ax[0], emb_orig, "t-SNE of the original space"),
        (ax[1], emb_repr, "t-SNE of the representation space"),
    ]:
        axis.set_title(title)
        # the two background classes are >99% of the points; drawn opaque they
        # saturate the panel and bury the signal, so they are faded and the
        # signal is drawn on top.
        axis.scatter(emb[is_3b, 0], emb[is_3b, 1], label=r"$3b$",
                     s=1.5, alpha=0.22, linewidths=0, rasterized=True, zorder=1)
        axis.scatter(emb[is_bg4b, 0], emb[is_bg4b, 1], label=r"Background $4b$",
                     s=1.5, alpha=0.22, linewidths=0, rasterized=True, zorder=2)
        if contours:
            signal_density_contours(axis, emb, is_signal)
        axis.scatter(emb[is_signal, 0], emb[is_signal, 1], label=r"$\mathrm{HH} \to 4b$",
                     s=34, edgecolors="black", linewidths=0.35, rasterized=True, zorder=3)
        leg = axis.legend(loc="lower right", framealpha=0.9)
        for handle in leg.legend_handles:
            handle.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=raster_dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n-points", type=int, default=20000)
    parser.add_argument("--tag", default="scan")
    parser.add_argument("--raster-dpi", type=int, default=200)
    parser.add_argument("--contours", action="store_true",
                        help="overlay signal-density contours (off by default)")
    parser.add_argument("--repr-members", type=int, default=1,
                        help="concatenate this many ensemble members' representations")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tsne_kwargs = dict(n_components=2, perplexity=20, learning_rate="auto",
                       init="pca", n_iter=500, early_exaggeration=12,
                       random_state=42, metric="euclidean")

    rows = []
    for seed in args.seeds:
        started = time.time()
        try:
            payload = embed_seed(
                seed, args.n_points, tsne_kwargs, device, n_members=args.repr_members)
            masks = payload["masks"]
            emb_orig, emb_repr = payload["emb_orig"], payload["emb_repr"]
            weights, space = payload["weights"], payload["space"]
        except Exception as error:                      # keep the scan going
            print(f"seed {seed}: FAILED ({type(error).__name__}: {error})", flush=True)
            continue
        is_signal = masks["is_signal"]
        rng = np.random.default_rng(1234)
        path = os.path.join(OUT_DIR, f"tsne_{args.tag}_seed{seed}.pdf")
        draw(seed, masks, emb_orig, emb_repr, path, raster_dpi=args.raster_dpi,
             contours=args.contours)
        row = {
            "seed": seed,
            "repr_members": args.repr_members,
            "n_points": len(is_signal),
            "n_signal": int(is_signal.sum()),
            "signal_frac": float(is_signal.mean()),
            # full-dimensional spaces -- the robust numbers
            "space_raw_w": space["raw"]["lift_w"],
            "space_repr_w": space["repr"]["lift_w"],
            "space_raw": space["raw"]["lift"],
            "space_repr": space["repr"]["lift"],
            # 2-D embedding -- what the reader actually sees
            "emb_orig_w": knn_signal_lift(emb_orig, is_signal, weights=weights),
            "emb_repr_w": knn_signal_lift(emb_repr, is_signal, weights=weights),
            "disp_orig": dispersion_ratio(emb_orig, is_signal, rng),
            "disp_repr": dispersion_ratio(emb_repr, is_signal, rng),
            "seconds": round(time.time() - started, 1),
            "figure": path,
        }
        row["space_gain_w"] = row["space_repr_w"] / row["space_raw_w"] if row["space_raw_w"] else float("nan")
        rows.append(row)
        print(f"seed {seed}: space repr/raw (weighted) = {row['space_repr_w']:.2f}/"
              f"{row['space_raw_w']:.2f} = {row['space_gain_w']:.2f}x   "
              f"| 2-D emb repr(w)={row['emb_repr_w']:.2f} ({row['seconds']}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"tsne_{args.tag}_summary.csv"), index=False)

    if rows:
        table = pd.DataFrame(rows).sort_values("space_gain_w", ascending=False)
        print("\n=== ranked by weighted kNN lift gain in the FULL representation space ===")
        print(table[["seed", "n_signal", "space_raw_w", "space_repr_w", "space_gain_w",
                     "space_raw", "space_repr",
                     "emb_orig_w", "emb_repr_w", "disp_repr"]].to_string(index=False))
        print(f"\nbest seed: {int(table.iloc[0].seed)} -> {table.iloc[0].figure}")


if __name__ == "__main__":
    main()
