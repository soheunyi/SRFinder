"""Generate the three draft power figures from continuous-affine results only."""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same font and label sizes as the draft figure notebooks, so the power plots
# match the rest of the manuscript figures and the imsart body font.
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{newtxtext}\usepackage{newtxmath}"
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["figure.labelsize"] = 30
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 20
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/refit_bootstrap/continuous_affine_full_v1"
SUMMARY = OUT / "power_figure_summary.csv"
FIGURE_DIR = REPO / "figures"
CSV_DIR = REPO / "notebooks/draft/csv"
BACKUP_DIR = OUT / "pre_continuous_affine_power_outputs"
SIGNALS = {
    "HH4b": "CR_fvt_training_ensemble_max",
    "HH4b_400": "CR_fvt_training_ensemble_max_HH4b_400",
    "ZH4b": "CR_fvt_training_ensemble_max_ZH4b",
}


def backup_once(path: Path) -> None:
    if not path.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / path.name
    if not backup.exists():
        shutil.copy2(path, backup)


def plot_power(data: pd.DataFrame, output: Path) -> None:
    data = data.copy().sort_values(["signal_ratio", "SR_size"])
    signal_levels = np.sort(data.signal_ratio.unique())
    sr_levels = np.sort(data.SR_size.unique())
    x = np.arange(len(signal_levels), dtype=float)
    width = 0.8 / len(sr_levels)
    offsets = (np.arange(len(sr_levels)) - (len(sr_levels) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for index, sr_size in enumerate(sr_levels):
        subset = data[data.SR_size == sr_size].sort_values("signal_ratio")
        y = 100 * subset.rejection_rate.to_numpy()
        lower = 100 * subset.ci_lower95.to_numpy()
        upper = 100 * subset.ci_upper95.to_numpy()
        ax.bar(x + offsets[index], y, width=width, label=f"{sr_size:g}", alpha=0.9)
        ax.errorbar(
            x + offsets[index],
            y,
            yerr=np.vstack([y - lower, upper - y]),
            fmt="none",
            capsize=3,
            linewidth=1,
            color="k",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"${value:g}$" for value in signal_levels])
    ax.set_xlabel(r"Signal Ratio ($\epsilon$)")
    ax.set_ylabel(r"Rejection Rate ($\%$)")
    ax.set_ylim(0, 105)
    ax.legend(
        title="SR Size",
        ncols=min(2, len(sr_levels)),
        fontsize=20,
        title_fontsize=20,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.axhline(5, color="red", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    null_rows = summary[
        (summary.experiment_name == SIGNALS["HH4b"])
        & (summary.signal_ratio == 0.0)
    ].copy()
    if len(null_rows) != 4:
        raise RuntimeError(f"Expected four shared null cells, found {len(null_rows)}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    for signal_name, experiment_name in SIGNALS.items():
        data = summary[summary.experiment_name == experiment_name].copy()
        if not (data.signal_ratio == 0.0).any():
            data = pd.concat([null_rows, data], ignore_index=True)
        cell_sizes = data.groupby(["signal_ratio", "SR_size"]).n.first()
        if not (cell_sizes == 100).all():
            raise RuntimeError(f"{signal_name}: expected 100 experiments per cell")

        csv_path = CSV_DIR / f"hypothesis_testing_{signal_name}_noise_scale=2.0.csv"
        figure_path = FIGURE_DIR / f"power_plot_{signal_name}_noise_scale=2.0.pdf"
        backup_once(csv_path)
        backup_once(figure_path)
        pivot = data.pivot(
            index="SR_size", columns="signal_ratio", values="rejection_rate"
        ).sort_index().sort_index(axis=1)
        pivot.columns = pd.MultiIndex.from_product(
            [["reject_null_correction"], pivot.columns],
            names=[None, "signal_ratio"],
        )
        pivot.to_csv(csv_path)
        plot_power(data, figure_path)
        shutil.copy2(csv_path, OUT / csv_path.name)
        shutil.copy2(figure_path, OUT / figure_path.name)
        print(f"generated {csv_path}")
        print(f"generated {figure_path}")


if __name__ == "__main__":
    main()
