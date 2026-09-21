from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta


REPO = Path("/home/export/soheuny/SRFinder/soheun")
OUTPUT_DIR = REPO / "data/refit_bootstrap/draft_eta2_logitcap10_v1"
SUMMARY_CSV = OUTPUT_DIR / "draft_eta2_old_vs_refit_summary.csv"
BACKUP_DIR = OUTPUT_DIR / "pre_refit_power_figures"
FIGURE_DIR = REPO / "figures"
CSV_DIR = REPO / "notebooks/draft/csv"

SIGNALS = {
    "HH4b": "CR_fvt_training_ensemble_max",
    "HH4b_400": "CR_fvt_training_ensemble_max_HH4b_400",
    "ZH4b": "CR_fvt_training_ensemble_max_ZH4b",
}


def clopper_pearson_ci(p_hat, n_trials, alpha=0.05):
    successes = int(round(float(p_hat) * n_trials))
    lower = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, n_trials - successes + 1)
    upper = 1.0 if successes == n_trials else beta.ppf(1 - alpha / 2, successes + 1, n_trials - successes)
    return lower, upper


def plot_power(data, output_path):
    data = data.copy().sort_values(["signal_ratio", "SR_size"])
    intervals = data.apply(
        lambda row: clopper_pearson_ci(row["new_power"], int(row["n"])), axis=1
    )
    data["ci_lower"] = [interval[0] for interval in intervals]
    data["ci_upper"] = [interval[1] for interval in intervals]
    for column in ["new_power", "ci_lower", "ci_upper"]:
        data[column] *= 100

    signal_levels = np.sort(data["signal_ratio"].unique())
    sr_levels = np.sort(data["SR_size"].unique())
    x = np.arange(len(signal_levels), dtype=float)
    group_width = 0.8
    bar_width = group_width / len(sr_levels)
    offsets = (np.arange(len(sr_levels)) - (len(sr_levels) - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for index, sr_size in enumerate(sr_levels):
        subset = data[data["SR_size"] == sr_size].sort_values("signal_ratio")
        y = subset["new_power"].to_numpy()
        yerr = np.vstack(
            [
                y - subset["ci_lower"].to_numpy(),
                subset["ci_upper"].to_numpy() - y,
            ]
        )
        ax.bar(
            x + offsets[index],
            y,
            width=bar_width,
            label=f"{sr_size:g}",
            alpha=0.9,
        )
        ax.errorbar(
            x + offsets[index],
            y,
            yerr=yerr,
            fmt="none",
            capsize=3,
            linewidth=1,
            color="k",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"${value:g}$" for value in signal_levels])
    ax.set_xlabel(r"Signal Ratio ($\epsilon$)")
    ax.set_ylabel(r"Power ($\%$)")
    ax.set_ylim(0, 105)
    ax.legend(
        title="SR Size",
        ncols=min(2, len(sr_levels)),
        fontsize=15,
        title_fontsize=15,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.axhline(5, color="red", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def backup_once(path):
    if not path.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / path.name
    if not backup.exists():
        shutil.copy2(path, backup)


summary = pd.read_csv(SUMMARY_CSV)
null_rows = summary[
    (summary["experiment_name"] == SIGNALS["HH4b"])
    & (summary["signal_ratio"] == 0.0)
].copy()
if len(null_rows) != 4:
    raise RuntimeError(f"Expected four shared null rows, found {len(null_rows)}")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)
for signal_name, experiment_name in SIGNALS.items():
    data = summary[summary["experiment_name"] == experiment_name].copy()
    if not (data["signal_ratio"] == 0.0).any():
        data = pd.concat([null_rows, data], ignore_index=True)
    cell_sizes = data.groupby(["signal_ratio", "SR_size"])["n"].first()
    if not (cell_sizes == 100).all():
        raise RuntimeError(
            f"{signal_name}: expected 100 experiments in every plotted cell"
        )

    csv_path = CSV_DIR / f"hypothesis_testing_{signal_name}_noise_scale=2.0.csv"
    figure_path = FIGURE_DIR / f"power_plot_{signal_name}_noise_scale=2.0.pdf"
    backup_once(csv_path)
    backup_once(figure_path)

    pivot = data.pivot(index="SR_size", columns="signal_ratio", values="new_power")
    pivot = pivot.sort_index().sort_index(axis=1)
    pivot.columns = pd.MultiIndex.from_product(
        [["reject_null_correction"], pivot.columns], names=[None, "signal_ratio"]
    )
    pivot.to_csv(csv_path)
    plot_power(data, figure_path)
    print(f"generated {csv_path}")
    print(f"generated {figure_path}")
