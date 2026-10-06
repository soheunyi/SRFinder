"""Summarize the three tests (composite, fixed, refit) on the observed 3b range.

Writes, per grid (full, eta_inf), to three_tests_obsrange_v1/<grid>/:
  detailed.csv                one row per config
  summary.csv                 rejection rate per experiment x eta x signal x SR, with
                              Wilson 95% intervals, for the three tests and the draft
  null_rate_by_eta_sr.png     null rejection rate per eta and SR (HH4b)
  power_<experiment>.png      eta = 2 power against signal, one panel per SR
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/export/soheuny/SRFinder/soheun/data/refit_bootstrap/three_tests_obsrange_v1")
EXPECTED = {"full": 12000, "eta_inf": 2000}
KEYS = ["experiment", "noise_scale", "signal_ratio", "sr_size"]
TESTS = ["composite", "fixed", "refit", "draft"]
LABELS = {"composite": "1 composite (max p over t)", "fixed": "2 fixed t-hat, no refit", "refit": "3 refit every replicate", "draft": "draft (composite on [cut, 10])"}
COLORS = {"composite": "tab:green", "fixed": "tab:blue", "refit": "tab:red", "draft": "0.5"}


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def load(grid: str) -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / grid / "results").glob("*.pkl")):
        with path.open("rb") as handle:
            r = pickle.load(handle)
        rows.append(
            {
                "hash": r["hash"],
                "experiment": r["experiment_name"].replace("CR_fvt_training_ensemble_max", "HH4b").replace("HH4b_", "").replace("400", "HH4b_400"),
                "noise_scale": r["noise_scale"],
                "signal_ratio": r["signal_ratio"],
                "sr_size": r["sr_size"],
                "seed": r["seed"],
                "for_noise_table": r.get("for_noise_table"),
                "for_power_figures": r.get("for_power_figures"),
                "has_signed_4b": r["has_signed_4b"],
                "t_hat": r["t_hat"],
                "t_hat_side": r["t_hat_side"],
                "p_composite": r["composite"]["p_value"],
                "p_fixed": r["fixed"]["p_value"],
                "p_refit": r["refit"]["p_value"],
                "p_draft": r["draft_p_cut_to_10"],
                "refit_matches_previous": r["refit"]["p_value"] == r["previous_refit_p"],
            }
        )
    d = pd.DataFrame(rows)
    for t in TESTS:
        d[t] = d[f"p_{t}"] <= 0.05
    return d


def summarize(grid: str) -> None:
    d = load(grid)
    out = ROOT / grid
    print(f"== {grid}: rows={len(d)}/{EXPECTED[grid]}  refit p equals earlier run: {int(d.refit_matches_previous.sum())}/{len(d)}  "
          f"edge fits: {d.t_hat_side.value_counts().to_dict()}  signed-4b configs: {int(d.has_signed_4b.sum())}")
    g = d.groupby(KEYS)
    s = g.size().rename("n").to_frame()
    for t in TESTS:
        k = g[t].sum()
        s[t] = k / s["n"]
        lo, hi = wilson(k.values, s["n"].values)
        s[f"{t}_lo"], s[f"{t}_hi"] = lo, hi
    s = s.reset_index()
    d.to_csv(out / "detailed.csv", index=False)
    s.to_csv(out / "summary.csv", index=False)

    null = d[d.signal_ratio == 0]
    print("NULL rejections per 100 (HH4b), rows eta, columns SR:")
    for t in TESTS:
        table = null.pivot_table(index="noise_scale", columns="sr_size", values=t, aggfunc="sum").astype(int)
        k, n = int(null[t].sum()), len(null)
        lo, hi = wilson(np.array([k]), np.array([n]))
        print(f"  {LABELS[t]}: pooled {k}/{n} = {100 * k / n:.1f}% (95% CI {100 * lo[0]:.1f}-{100 * hi[0]:.1f}), max cell {table.values.max()}")
        print("    " + table.to_string().replace("\n", "\n    "))
    print("  null rejection % by fitted-t side:", (100 * null.groupby("t_hat_side")[TESTS].mean()).round(1).to_dict(orient="index"))

    power = s[s.signal_ratio > 0]
    print("POWER (mean over SR) per experiment, eta, signal:")
    print(power.groupby(["experiment", "noise_scale", "signal_ratio"])[TESTS].mean().round(3).to_string())
    for a, b in (("fixed", "composite"), ("refit", "composite"), ("refit", "fixed"), ("composite", "draft")):
        diff = power[a] - power[b]
        print(f"  {a} vs {b}: higher by >=0.05 in {int((diff >= 0.05).sum())}, lower by >=0.05 in {int((diff <= -0.05).sum())}, of {len(power)} power cells")

    # figures
    etas = sorted(null.noise_scale.unique())
    srs = sorted(null.sr_size.unique())
    if len(null):
        fig, axes = plt.subplots(1, len(etas), figsize=(3.6 * len(etas), 3.2), squeeze=False, sharey=True)
        for ax, eta in zip(axes.flat, etas):
            cell = s[(s.signal_ratio == 0) & (s.noise_scale == eta)]
            for j, t in enumerate(TESTS):
                x = np.arange(len(cell)) + (j - 1.5) * 0.12
                ax.errorbar(x, cell[t], yerr=[np.clip(cell[t] - cell[f"{t}_lo"], 0, None), np.clip(cell[f"{t}_hi"] - cell[t], 0, None)], fmt="o", ms=4, color=COLORS[t], label=LABELS[t])
            ax.axhline(0.05, color="k", lw=0.8, ls=":")
            ax.set_xticks(np.arange(len(cell)), [f"{v:g}" for v in cell.sr_size])
            ax.set_xlabel("SR size")
            ax.set_title(f"eta = {eta:g}")
        axes.flat[0].set_ylabel("null rejection rate")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"HH4b null rejection rate (alpha = 0.05, 100 seeds per cell, Wilson 95% CI), grid {grid}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "null_rate_by_eta_sr.png", dpi=130)
        plt.close(fig)
    target_eta = 2.0 if grid == "full" else float("inf")
    for experiment in sorted(power.experiment.unique()):
        cells = s[(s.experiment == experiment) & (s.noise_scale == target_eta)]
        if cells.empty:
            continue
        fig, axes = plt.subplots(1, len(srs), figsize=(3.6 * len(srs), 3.2), squeeze=False, sharey=True)
        for ax, sr in zip(axes.flat, srs):
            cell = cells[cells.sr_size == sr].sort_values("signal_ratio")
            for t in TESTS:
                ax.errorbar(cell.signal_ratio, cell[t], yerr=[np.clip(cell[t] - cell[f"{t}_lo"], 0, None), np.clip(cell[f"{t}_hi"] - cell[t], 0, None)],
                            marker="o", ms=3.5, lw=1.2, color=COLORS[t], label=LABELS[t], capsize=2)
            ax.axhline(0.05, color="k", lw=0.8, ls=":")
            ax.set_title(f"SR size {sr:g}")
            ax.set_xlabel("signal ratio")
        axes.flat[0].set_ylabel("rejection rate")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"{experiment}, eta = {target_eta:g}: rejection rate vs signal ratio (100 seeds per point)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"power_{experiment}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    for grid in sys.argv[1:] or ["full", "eta_inf"]:
        summarize(grid)
