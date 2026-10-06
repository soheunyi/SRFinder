"""eta = infinity version of plot_needed_correction.py (null, SR 0.2, affine fit only).

Plot the correction the data needs against the correction the test fits.

For HH4b, eta = 2, SR 0.2: the needed correction is
    r(psi) = (4b density) / (reweighted 3b density),
both normalized to total weight one, as the KS test compares them. The x axis is
u = reweighted-3b CDF at psi (the KS test depends only on the order of scores, and
on this axis 3b weight is uniform; the sparse high-score tail is the last few
percent of u). r is shown as 30 equal-width u bins with error bars and as a
Gaussian-kernel ratio in u (bandwidth 0.025). A lower panel shows the CDF gap the
KS test sees, corrected 3b CDF minus 4b CDF, before correction and after each fit. Overlays: the affine L-infinity literal fit h(psi) (observed 3b
range) and the free-quadratic fit, both scaled so that sum(base3 * h) = 1, i.e.
on the same scale as r.

Cases: null (all affine rejections and 5 non-rejections), eps = 0.02 (split by
the free-quadratic decision; the affine test rejects every seed), and eps =
0.0075 (split by the affine decision). Writes one PNG per case and a CSV of the
curves to OUT.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/export/soheuny/SRFinder/soheun")
FULL = REPO / "data/refit_bootstrap/centered_poisson_obsrange_linf_refit_eta_inf_v1"
FREE = REPO / "data/refit_bootstrap/centered_poisson_free_quadratic_eta2_v1"
OUT = REPO / "data/refit_bootstrap/needed_correction_plots_v2_eta_inf"
EXPERIMENT = "CR_fvt_training_ensemble_max"
ETA = float("inf")
SR = 0.2
N_BINS = 30
GRID = 300

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "run_files"))
os.chdir(REPO)
from run_files.run_continuous_affine_eta_inf import load_arrays_and_cutoff


def bernstein(x: np.ndarray) -> np.ndarray:
    return np.vstack([(1 - x) ** 2, 2 * x * (1 - x), x**2])


def h_affine_at(t_hat: float, s3, w3, psi):
    lo, hi = s3.min(), s3.max()
    x = (s3 - lo) / (hi - lo)
    base3 = w3 / w3.sum()
    m_minus, m_plus = np.sum(base3 * (1 - x)), np.sum(base3 * x)
    xp = (np.asarray(psi) - lo) / (hi - lo)
    return (1 - t_hat) * (1 - xp) / m_minus + t_hat * xp / m_plus


def h_quadratic_at(c_hat, s3, w3, psi):
    lo, hi = s3.min(), s3.max()
    base3 = w3 / w3.sum()
    c = np.asarray(c_hat)
    scale = np.sum(base3 * (c @ bernstein((s3 - lo) / (hi - lo))))
    return (c @ bernstein((np.asarray(psi) - lo) / (hi - lo))) / scale


def analyse(s3, w3, s4, w4, t_hat, c_hat):
    p3 = w3 / w3.sum()
    p4 = w4 / w4.sum()
    order = np.argsort(s3)
    s3_sorted = s3[order]
    cum3 = np.cumsum(p3[order])

    def u_of(psi):
        return np.interp(psi, s3_sorted, cum3, left=0.0, right=1.0)

    u3, u4 = u_of(s3), u_of(s4)
    edges = np.linspace(0, 1, N_BINS + 1)
    i3 = np.clip(np.searchsorted(edges, u3, side="right") - 1, 0, N_BINS - 1)
    i4 = np.clip(np.searchsorted(edges, u4, side="right") - 1, 0, N_BINS - 1)
    a, b = np.bincount(i4, p4, N_BINS), np.bincount(i3, p3, N_BINS)
    va, vb = np.bincount(i4, p4**2, N_BINS), np.bincount(i3, p3**2, N_BINS)
    ratio = a / b
    err = ratio * np.sqrt(va / a**2 + vb / b**2)
    centers = 0.5 * (edges[:-1] + edges[1:])

    grid = np.linspace(0.005, 0.995, GRID)
    bw = 0.025

    def kd(u, p):
        return np.array([np.sum(p * np.exp(-0.5 * ((u - g) / bw) ** 2)) for g in grid])

    smooth = kd(u4, p4) / kd(u3, p3)
    psi_grid = np.interp(grid, cum3, s3_sorted)
    h_aff = h_affine_at(t_hat, s3, w3, psi_grid)
    h_quad = h_quadratic_at(c_hat, s3, w3, psi_grid)

    # CDF gaps on the pooled support, reported at u of each support point.
    support = np.unique(np.concatenate((s3, s4)))
    def cdf(scores, probs):
        idx = np.searchsorted(support, scores)
        return np.cumsum(np.bincount(idx, probs, support.size))
    f4 = cdf(s4, p4)
    gaps = {}
    for name, h in (("none", np.ones_like(s3)), ("affine", h_affine_at(t_hat, s3, w3, s3)), ("quadratic", h_quadratic_at(c_hat, s3, w3, s3))):
        q = p3 * h
        gaps[name] = cdf(s3, q / q.sum()) - f4
    return centers, ratio, err, grid, smooth, h_aff, h_quad, u_of(support), gaps


def select() -> list[tuple[str, dict]]:
    full = pd.read_csv(FULL / "obsrange_linf_refit_eta_inf_detailed.csv")
    cell = full[(full.experiment_name == EXPERIMENT) & np.isinf(full.noise_scale) & (full.sr_size == SR) & (full.signal_ratio == 0)]
    picks = [("null_eta_inf", r) for _, r in cell[cell.linf_literal].iterrows()]
    picks += [("null_eta_inf", r) for _, r in cell[~cell.linf_literal].sample(12 - len(picks), random_state=0).iterrows()]
    return picks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    picks = select()
    curves = []
    panels: dict[str, list] = {}
    for case, row in picks:
        with (FULL / "results" / f"{row['hash']}.pkl").open("rb") as handle:
            full = pickle.load(handle)
        load_row = {k: full[k] for k in ("hash", "experiment_name", "noise_scale", "signal_ratio", "sr_size", "seed")}
        _t, s3, w3, s4, w4, *_ = load_arrays_and_cutoff(load_row)
        centers, ratio, err, grid, smooth, h_aff, h_quad, u_support, gaps = analyse(
            s3, w3, s4, w4, full["linf_literal"]["observed_t"], [1.0, 1.0, 1.0]
        )
        info = {
            "seed": int(row["seed"]),
            "p_affine": full["linf_literal"]["p_value"],
            "p_free_quad": float("nan"),
            "p_draft": full["draft_p_value"],
        }
        panels.setdefault(case, []).append((info, centers, ratio, err, grid, smooth, h_aff, h_quad, u_support, gaps))
        for g, sm, ha, hq in zip(grid, smooth, h_aff, h_quad):
            curves.append({"case": case, **info, "u": g, "kernel_ratio": sm, "h_affine": ha, "h_free_quad": hq})
        print(case, info, flush=True)

    pd.DataFrame(curves).to_csv(OUT / "needed_correction_curves.csv", index=False)

    for case, items in panels.items():
        n = len(items)
        cols = 4 if n > 6 else 3
        rows_n = int(np.ceil(n / cols))
        fig = plt.figure(figsize=(4.4 * cols, 4.6 * rows_n))
        outer = fig.add_gridspec(rows_n, cols, hspace=0.45, wspace=0.28)
        ylim = (0.85, 1.15)
        for k, (info, centers, ratio, err, grid, smooth, h_aff, h_quad, u_support, gaps) in enumerate(items):
            inner = outer[k // cols, k % cols].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
            ax = fig.add_subplot(inner[0])
            gx = fig.add_subplot(inner[1], sharex=ax)
            ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
            ax.errorbar(centers, ratio, yerr=err, fmt="o", ms=2.5, color="k", lw=0.7, label="needed (bins)")
            ax.plot(grid, smooth, color="k", lw=1.4, label="needed (kernel)")
            ax.plot(grid, h_aff, color="tab:blue", lw=1.6, label="affine fit")
            ax.set_ylim(*ylim)
            ax.tick_params(labelbottom=False, labelsize=7)
            rejected = info["p_affine"] <= 0.05
            ax.set_title(f"seed {info['seed']}  p_aff={info['p_affine']:.3f}  p_draft={info['p_draft']:.3f}",
                         fontsize=8.5, color="tab:red" if rejected else "k")
            for name, color, ls in (("none", "0.6", "-"), ("affine", "tab:blue", "-")):
                g = gaps[name]
                gx.plot(u_support, g * 1e3, color=color, lw=1.0, ls=ls)
                j = int(np.argmax(np.abs(g)))
                gx.plot(u_support[j], g[j] * 1e3, "o", color=color, ms=3.5)
            gx.axhline(0, color="0.5", lw=0.6)
            gx.set_xlim(0, 1)
            gx.set_xlabel("u = reweighted-3b CDF at psi", fontsize=7.5)
            gx.set_ylabel("CDF gap x1e3", fontsize=7)
            gx.tick_params(labelsize=7)
            if k == 0:
                ax.legend(fontsize=6.5, loc="upper left")
        fig.suptitle(
            f"HH4b, eta=inf, SR 0.2, null. Top: needed correction r(u) = 4b / reweighted 3b, with the affine fit. "
            "Bottom: CDF gap (grey none, blue affine; dot = largest gap). Red title = affine test rejects.",
            fontsize=9.5,
        )
        fig.savefig(OUT / f"needed_correction_{case}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")), flush=True)


if __name__ == "__main__":
    main()
