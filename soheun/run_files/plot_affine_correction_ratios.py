"""Plot seed distributions of fitted affine endpoint correction ratios."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_REPO = Path(
    os.environ.get("SRFINDER_DATA_REPO", "/home/export/soheuny/SRFinder/soheun")
)
LOCAL_DIR = DATA_REPO / "data/refit_bootstrap/centered_poisson_affine_null_v1"
SYMMETRIC_DIR = (
    DATA_REPO / "data/refit_bootstrap/centered_poisson_affine_symmetric16_null_v1"
)
ETAS = [0.5, 1.0, 2.0, np.inf]
SR_SIZES = [0.05, 0.20]
LOG_BINS = np.linspace(-3.0, 3.0, 31)


def ratio_data(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    result = frame.copy()
    if family == "local [L,10]":
        numerator = result["correction_value_L"].to_numpy(float)
        denominator = result["correction_value_U"].to_numpy(float)
    elif family == "global [-16,16]":
        numerator = result["correction_value_minus16"].to_numpy(float)
        denominator = result["correction_value_plus16"].to_numpy(float)
    else:
        raise ValueError(f"unknown family {family}")
    ratio = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator != 0,
    )
    ratio[(denominator == 0) & (numerator > 0)] = np.inf
    ratio[(numerator == 0) & (denominator > 0)] = 0.0
    result["correction_ratio_L_over_U"] = ratio
    result["family"] = family
    return result


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, eta, sr), group in frame.groupby(
        ["family", "noise_scale", "sr_size"], sort=True
    ):
        values = group["correction_ratio_L_over_U"].to_numpy(float)
        finite_positive = values[np.isfinite(values) & (values > 0)]
        rows.append(
            {
                "family": family,
                "noise_scale": eta,
                "sr_size": sr,
                "n": len(values),
                "finite_positive": len(finite_positive),
                "zero_ratio": int(np.count_nonzero(values == 0)),
                "infinite_ratio": int(np.count_nonzero(np.isposinf(values))),
                "median_finite_ratio": (
                    float(np.median(finite_positive))
                    if finite_positive.size
                    else np.nan
                ),
                "q25_finite_ratio": (
                    float(np.quantile(finite_positive, 0.25))
                    if finite_positive.size
                    else np.nan
                ),
                "q75_finite_ratio": (
                    float(np.quantile(finite_positive, 0.75))
                    if finite_positive.size
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def eta_mask(series: pd.Series, eta: float) -> np.ndarray:
    values = series.to_numpy(float)
    return np.isinf(values) if np.isinf(eta) else values == eta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", action="store_true")
    args = parser.parse_args()

    local = pd.read_csv(LOCAL_DIR / "centered_poisson_affine_null_detailed.csv")
    frames = [ratio_data(local, "local [L,10]")]
    output_dir = LOCAL_DIR
    if args.comparison:
        symmetric_path = (
            SYMMETRIC_DIR / "centered_poisson_affine_symmetric16_null_detailed.csv"
        )
        if not symmetric_path.exists():
            raise FileNotFoundError("symmetric campaign has not been aggregated")
        frames.append(
            ratio_data(pd.read_csv(symmetric_path), "global [-16,16]")
        )
        output_dir = SYMMETRIC_DIR
    data = pd.concat(frames, ignore_index=True)
    summary = summarize(data)
    summary.to_csv(output_dir / "correction_ratio_summary.csv", index=False)

    colors = {"local [L,10]": "#1f77b4", "global [-16,16]": "#d62728"}
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    for row_index, sr in enumerate(SR_SIZES):
        for column_index, eta in enumerate(ETAS):
            ax = axes[row_index, column_index]
            annotations = []
            for family, family_data in data.groupby("family", sort=False):
                selected = family_data[
                    eta_mask(family_data["noise_scale"], eta)
                    & (family_data["sr_size"] == sr)
                ]
                values = selected["correction_ratio_L_over_U"].to_numpy(float)
                finite = values[np.isfinite(values) & (values > 0)]
                log_values = np.clip(np.log10(finite), LOG_BINS[0], LOG_BINS[-1])
                ax.hist(
                    log_values,
                    bins=LOG_BINS,
                    histtype="step",
                    linewidth=1.7,
                    color=colors[family],
                    label=family,
                )
                annotations.append(
                    f"{family}: 0={np.count_nonzero(values == 0)}, "
                    f"inf={np.count_nonzero(np.isposinf(values))}"
                )
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
            eta_label = r"\infty" if np.isinf(eta) else f"{eta:g}"
            ax.set_title(rf"$\eta={eta_label}$")
            ax.text(
                0.02,
                0.97,
                "\n".join(annotations),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )
            if column_index == 0:
                ax.set_ylabel(rf"$s_{{\rm SR}}={sr:g}$\ncount")
            if row_index == 1:
                ax.set_xlabel(r"$\log_{10}\{h(L)/h(U)\}$")
            ax.grid(alpha=0.2)
    axes[0, -1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Fitted affine correction ratios across 100 null seeds\n"
        "finite positive ratios shown; exact zero/infinite boundary masses annotated"
    )
    fig.tight_layout()
    stem = "correction_ratio_histograms_comparison" if args.comparison else "correction_ratio_histograms_local"
    fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    print(summary.to_string(index=False))
    print(f"saved={output_dir / (stem + '.png')}")


if __name__ == "__main__":
    main()
