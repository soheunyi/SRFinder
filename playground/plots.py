# Drawing weighted histograms


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_prob_weighted_histogram1d(
    probs_4b: np.ndarray,
    plot_feature_arr: np.ndarray,
    labels_4b: np.ndarray,
    n_bins: int = 7,
    sample_weights: np.ndarray = None,
    output_file: str = None,
    ylim: tuple[float, float] = None,
):
    if sample_weights is None:
        sample_weights = np.ones_like(plot_feature_arr)
    assert (
        len(probs_4b) == len(plot_feature_arr) == len(labels_4b) == len(sample_weights)
    )
    weights = probs_4b / (1 - probs_4b)
    weights_3b = weights[labels_4b == 0]
    samples_3b_np = plot_feature_arr[labels_4b == 0]
    samples_4b_np = plot_feature_arr[labels_4b == 1]
    samples_3b_weights = sample_weights[labels_4b == 0]
    samples_4b_weights = sample_weights[labels_4b == 1]

    w_hist_3b, x_edges = np.histogram(
        samples_3b_np, bins=n_bins, weights=samples_3b_weights * weights_3b
    )
    w_sq_hist_3b, _ = np.histogram(
        samples_3b_np, bins=n_bins, weights=samples_3b_weights * weights_3b**2
    )

    hist_4b = np.zeros(n_bins)
    for i, (x_low, x_high) in enumerate(zip(x_edges[:-1], x_edges[1:])):
        falls_within = (samples_4b_np >= x_low) & (samples_4b_np < x_high)
        hist_4b[i] = np.sum(samples_4b_weights[falls_within])

    ratio_mean = hist_4b / w_hist_3b
    ratio_std = np.sqrt(
        hist_4b * (1 / w_hist_3b) ** 2 + w_sq_hist_3b * (hist_4b / w_hist_3b**2) ** 2
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    alpha = 0.05
    z = stats.norm.ppf(1 - alpha / 2)
    ratio_lb = ratio_mean.reshape(-1) - z * ratio_std.reshape(-1)
    ratio_ub = ratio_mean.reshape(-1) + z * ratio_std.reshape(-1)
    includes_one = (ratio_lb < 1) & (ratio_ub > 1)

    x_arr = np.arange(ratio_mean.size)
    y_arr = ratio_mean.reshape(-1)
    y_err_arr = z * ratio_std.reshape(-1)

    x_edge_centers = (x_edges[1:] + x_edges[:-1]) / 2

    # color = "green" if includes_one else "red"
    ax.errorbar(
        x_edge_centers[includes_one],
        y_arr[includes_one],
        yerr=y_err_arr[includes_one],
        fmt="o",
        capsize=5,
        color="green",
    )
    ax.errorbar(
        x_edge_centers[~includes_one],
        y_arr[~includes_one],
        yerr=y_err_arr[~includes_one],
        fmt="o",
        capsize=5,
        color="red",
    )
    ax.hlines(1, x_edges[0], x_edges[-1], color="red")
    ax.set_title(
        f"Approx. {100 * (1 - alpha)}% interval \nRegions that do not include 1: {x_edge_centers[~includes_one]}"
    )

    ax2 = ax.twinx()
    ax2.set_ylabel("Number of samples")
    ax2.hist(plot_feature_arr, bins=x_edges, alpha=0.25, color="gray")

    if ylim != None:
        ax.set_ylim(ylim)

    if output_file != None:
        plt.savefig(output_file)
    plt.show()
    plt.close()


def plot_prob_weighted_histogram2d(
    probs_4b: np.ndarray,
    df_val: pd.DataFrame,
    plot_features: list[str],
    n_bins: int = 7,
    output_file: str = None,
    ylim: tuple[float, float] = None,
):
    assert "target" in df_val.columns

    weights = probs_4b / (1 - probs_4b)
    weights = weights.reshape(-1)
    weights_3b = weights[df_val["target"] == 1]

    samples_3b_np = df_val.loc[df_val["target"] == 1, plot_features].values
    samples_4b_np = df_val.loc[df_val["target"] == 0, plot_features].values
    samples_3b_weights = df_val.loc[df_val["target"] == 1, "weight"].values
    samples_4b_weights = df_val.loc[df_val["target"] == 0, "weight"].values

    w_hist_3b, x_edges, y_edges = np.histogram2d(
        samples_3b_np[:, 0],
        samples_3b_np[:, 1],
        bins=n_bins,
        weights=samples_3b_weights * weights_3b,
    )
    w_sq_hist_3b, _, _ = np.histogram2d(
        samples_3b_np[:, 0],
        samples_3b_np[:, 1],
        bins=n_bins,
        weights=samples_3b_weights * weights_3b**2,
    )
    hist_4b, _, _ = np.histogram2d(
        samples_4b_np[:, 0],
        samples_4b_np[:, 1],
        range=[(x_edges[0], x_edges[-1]), (y_edges[0], y_edges[-1])],
        bins=n_bins,
        weights=samples_4b_weights,
    )
    ratio_mean = (hist_4b / w_hist_3b).T
    ratio_std = np.sqrt(
        hist_4b * (1 / w_hist_3b) ** 2 + w_sq_hist_3b * (hist_4b / w_hist_3b**2) ** 2
    ).T

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    alpha = 0.05
    z = stats.norm.ppf(1 - alpha / 2)
    ratio_lb = ratio_mean.reshape(-1) - z * ratio_std.reshape(-1)
    ratio_ub = ratio_mean.reshape(-1) + z * ratio_std.reshape(-1)
    includes_one = (ratio_lb < 1) & (ratio_ub > 1)

    x_arr = np.arange(ratio_mean.size)
    y_arr = ratio_mean.reshape(-1)
    y_err_arr = z * ratio_std.reshape(-1)

    # color = "green" if includes_one else "red"
    ax[0].errorbar(
        x_arr[includes_one],
        y_arr[includes_one],
        yerr=y_err_arr[includes_one],
        fmt="o",
        capsize=5,
        color="green",
    )
    ax[0].errorbar(
        x_arr[~includes_one],
        y_arr[~includes_one],
        yerr=y_err_arr[~includes_one],
        fmt="o",
        capsize=5,
        color="red",
    )
    ax[0].hlines(1, 0, ratio_mean.size, color="red")
    ax[0].set_title(
        f"Approx. {100 * (1 - alpha)}% interval \nRegions that do not include 1: {x_arr[~includes_one]}"
    )

    if ylim != None:
        ax[0].set_ylim(ylim)

    im = ax[1].imshow(
        ratio_mean,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        origin="lower",
    )
    # Text region number
    x_edge_centers = (x_edges[1:] + x_edges[:-1]) / 2
    y_edge_centers = (y_edges[1:] + y_edges[:-1]) / 2
    for i in range(ratio_mean.shape[1]):
        for j in range(ratio_mean.shape[0]):
            ax[1].text(
                x_edge_centers[j],
                y_edge_centers[i],
                f"{ratio_mean[i, j]:.2f}\n{ratio_mean.shape[1] * i + j}",
                ha="center",
                va="center",
                color="w",
            )
    plt.colorbar(im, ax=ax[1])
    if output_file != None:
        plt.savefig(output_file)
    plt.show()
    plt.close()


def calibration_plot(
    probs_est: np.ndarray,
    true_labels: np.ndarray,
    bins: int = 20,
    sample_weights: np.ndarray = None,
):
    if sample_weights is None:
        sample_weights = np.ones_like(probs_est)
    assert len(probs_est) == len(true_labels) == len(sample_weights)

    prob_min, prob_max = min(probs_est), max(probs_est)
    xs = np.linspace(prob_min, prob_max, bins + 1)
    probs_actual = np.zeros(bins)
    bincounts = np.zeros(bins)
    errors = np.zeros(bins)

    for i, (x_low, x_high) in enumerate(zip(xs[:-1], xs[1:])):
        falls_within = (probs_est >= x_low) & (probs_est < x_high)
        bincounts[i] = np.sum(sample_weights[falls_within])
        if bincounts[i] > 1:
            probs_actual[i] = (
                np.sum((sample_weights * true_labels)[falls_within]) / bincounts[i]
            )
            errors[i] = 1.96 * np.sqrt(
                probs_actual[i] * (1 - probs_actual[i]) / bincounts[i]
            )
        else:
            probs_actual[i] = np.nan
            errors[i] = np.nan

    x_arr = (xs[:-1] + xs[1:]) / 2
    fig, ax = plt.subplots(figsize=(5, 5))
    includes_theo = np.abs(probs_actual - x_arr) < errors
    ax.errorbar(
        x_arr[includes_theo],
        probs_actual[includes_theo],
        yerr=errors[includes_theo],
        fmt="o",
        capsize=4,
        markersize=4,
        color="green",
        label="Calibrated",
    )
    ax.errorbar(
        x_arr[~includes_theo],
        probs_actual[~includes_theo],
        yerr=errors[~includes_theo],
        fmt="o",
        capsize=4,
        markersize=4,
        color="red",
        label="Not calibrated",
    )
    # ideal line
    ax.plot(
        [prob_min, prob_max], [prob_min, prob_max], "k:", label="Perfectly calibrated"
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    # other values in the right axis
    ax2 = ax.twinx()
    ax2.set_ylabel("Number of samples")
    ax.legend()
    ax2.hist(probs_est, bins=bins, alpha=0.5, color="gray")
    # ax2.set_yscale("log")
    plt.show()
    plt.close()


def plot_cluster(
    q_repr,
    is_3b,
    is_bg4b,
    is_hh4b,
    weights,
    n_components=2,
    title="",
):

    assert q_repr.shape[1] == n_components

    if n_components == 1:
        # histogram
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        fig.suptitle(
            "\n".join(
                [
                    title,
                    "3b: {}, bg 4b: {}, HH 4b: {}".format(
                        np.sum(is_3b), np.sum(is_bg4b), np.sum(is_hh4b)
                    ),
                ]
            )
        )
        q_repr = q_repr.reshape(-1)
        bins_range = np.linspace(np.min(q_repr), np.max(q_repr), 50)
        hist_3b, _, _ = ax[0].hist(
            q_repr[is_3b],
            weights=weights[is_3b],
            bins=bins_range,
            label="bg 3b",
            linewidth=1,
            histtype="step",
            density=False,
        )
        hist_bg4b, _, _ = ax[0].hist(
            q_repr[is_bg4b],
            weights=weights[is_bg4b],
            bins=bins_range,
            label="bg 4b",
            linewidth=1,
            histtype="step",
            density=False,
        )
        hist_hh4b, _, _ = ax[0].hist(
            q_repr[is_hh4b],
            weights=weights[is_hh4b],
            bins=bins_range,
            label="HH 4b",
            linewidth=1,
            histtype="step",
            density=False,
        )
        ax[0].legend()
        ax[0].set_xlabel("cluster")
        # 4b / 3b ratio

        hist_4b = hist_bg4b + hist_hh4b
        ratio_mean = hist_4b / hist_3b
        ratio_std = np.sqrt(hist_4b * (1 / hist_3b) ** 2 + (hist_4b / hist_3b**2) ** 2)
        alpha = 0.05
        z = stats.norm.ppf(1 - alpha / 2)
        ratio_lb = ratio_mean.reshape(-1) - z * ratio_std.reshape(-1)
        ratio_ub = ratio_mean.reshape(-1) + z * ratio_std.reshape(-1)
        ax[1].plot(bins_range[:-1], ratio_mean, label="4b / 3b")
        ax[1].fill_between(bins_range[:-1], ratio_lb, ratio_ub, alpha=0.3)
        ax[1].legend()
        ax[1].set_xlabel("cluster")

        plt.show()
        plt.close()
    else:
        fig = go.Figure()
        fig.update_layout(width=600, height=600)
        fig.update_layout(hovermode=False)
        fig.update_layout(
            title="<br>".join(
                [
                    title,
                    "3b: {}, bg 4b: {}, HH 4b: {}".format(
                        np.sum(is_3b), np.sum(is_bg4b), np.sum(is_hh4b)
                    ),
                ]
            )
        )
        if n_components == 2:
            fig.add_trace(
                go.Scatter(
                    x=q_repr[is_3b, 0],
                    y=q_repr[is_3b, 1],
                    mode="markers",
                    name="bg 3b",
                    marker=dict(size=2, color="blue"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=q_repr[is_bg4b, 0],
                    y=q_repr[is_bg4b, 1],
                    mode="markers",
                    name="bg 4b",
                    marker=dict(size=2, color="orange"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=q_repr[is_hh4b, 0],
                    y=q_repr[is_hh4b, 1],
                    mode="markers",
                    name="HH 4b",
                    marker=dict(size=3, color="green"),
                )
            )
        elif n_components == 3:
            fig.add_trace(
                go.Scatter3d(
                    x=q_repr[is_3b, 0],
                    y=q_repr[is_3b, 1],
                    z=q_repr[is_3b, 2],
                    mode="markers",
                    name="bg 3b",
                    marker=dict(size=2, color="blue"),
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=q_repr[is_bg4b, 0],
                    y=q_repr[is_bg4b, 1],
                    z=q_repr[is_bg4b, 2],
                    mode="markers",
                    name="bg 4b",
                    marker=dict(size=2, color="orange"),
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=q_repr[is_hh4b, 0],
                    y=q_repr[is_hh4b, 1],
                    z=q_repr[is_hh4b, 2],
                    mode="markers",
                    name="HH 4b",
                    marker=dict(size=3, color="green"),
                )
            )

        fig.show()
