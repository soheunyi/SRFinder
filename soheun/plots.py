# Drawing weighted histograms


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytorch_lightning as pl

from events_data import EventsData


def plot_prob_weighted_histogram1d(
    probs_4b: np.ndarray,
    plot_feature_arr: np.ndarray,
    labels_4b: np.ndarray,
    n_bins: int = 7,
    sample_weights: np.ndarray = None,
    ylim: tuple[float, float] = None,
    show_plot: bool = True,
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

    if show_plot:
        plt.show()
        plt.close()
    else:
        return fig


def plot_prob_weighted_histogram2d(
    probs_4b: np.ndarray,
    df_val: pd.DataFrame,
    plot_features: list[str],
    n_bins: int = 7,
    ylim: tuple[float, float] = None,
    show_plot: bool = True,
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

    if show_plot:
        plt.show()
        plt.close()
    else:
        return fig


def calibration_plot(
    probs_est: np.ndarray,
    true_labels: np.ndarray,
    bins: int = 20,
    sample_weights: np.ndarray = None,
    show_plot: bool = True,
    title: str = "",
    ax=None,
):
    if sample_weights is None:
        sample_weights = np.ones_like(probs_est)
    assert len(probs_est) == len(true_labels) == len(sample_weights)

    prob_min, prob_max = min(probs_est), max(probs_est)
    if isinstance(bins, int):
        bins = np.linspace(prob_min, prob_max, bins + 1)

    probs_actual = np.zeros(len(bins) - 1)
    bincounts = np.zeros(len(bins) - 1)
    errors = np.zeros(len(bins) - 1)

    for i, (x_low, x_high) in enumerate(zip(bins[:-1], bins[1:])):
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

    x_arr = (bins[:-1] + bins[1:]) / 2

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.suptitle(title)
    else:
        fig = ax.get_figure()

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
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    # other values in the right axis
    ax2 = ax.twinx()
    ax2.set_ylabel("Number of samples")
    ax.legend()
    ax2.hist(probs_est, bins=bins, alpha=0.5, color="gray")
    # ax2.set_yscale("log")

    if show_plot:
        fig.show()
    else:
        return fig


def plot_cluster(
    q_repr: np.ndarray,
    events_data: EventsData,
    n_components=2,
    title="",
    show_plot=True,
    n_bins=50,
):
    # assert isinstance(events_data, EventsData)
    assert q_repr.shape[0] == len(events_data)
    is_3b = events_data.is_3b
    is_bg4b = events_data.is_bg4b
    is_signal = events_data.is_signal
    weights = events_data.weights

    assert q_repr.shape[1] == n_components

    if n_components == 1:
        # histogram
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
        fig.suptitle(
            "\n".join(
                [
                    title,
                    "3b: {}, bg 4b: {}, Signal: {}".format(
                        np.sum(is_3b), np.sum(is_bg4b), np.sum(is_signal)
                    ),
                ]
            )
        )
        q_repr = q_repr.reshape(-1)
        bins_range = np.linspace(np.min(q_repr), np.max(q_repr), n_bins)
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
        hist_signal, _, _ = ax[0].hist(
            q_repr[is_signal],
            weights=weights[is_signal],
            bins=bins_range,
            label="Signal",
            linewidth=1,
            histtype="step",
            density=False,
        )
        ax[0].legend()
        ax[0].set_xlabel("cluster")
        # 4b / 3b ratio

        hist_4b = hist_bg4b + hist_signal
        hist_all = hist_3b + hist_4b

        # ignore warnings
        alpha = 0.05
        l2a = np.log(2 / alpha)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_4b_mean_count = hist_4b / hist_all

            ratio_4b_mean_fvt = np.zeros_like(hist_4b)
            for i in range(len(hist_4b)):
                weights_bin = weights[
                    (q_repr >= bins_range[i]) & (q_repr < bins_range[i + 1])
                ]
                fvt_scores_bin = events_data.fvt_score[
                    (q_repr >= bins_range[i]) & (q_repr < bins_range[i + 1])
                ]
                ratio_4b_mean_fvt[i] = np.sum(weights_bin * fvt_scores_bin) / np.sum(
                    weights_bin
                )

            var_ratio_4b_count = ratio_4b_mean_count * (1 - ratio_4b_mean_count)
            var_ratio_4b_fvt = ratio_4b_mean_fvt * (1 - ratio_4b_mean_fvt)
            # bernstein style error
            ratio_4b_err_count = (
                4 * np.sqrt(var_ratio_4b_count * l2a / hist_all) + 4 * l2a / hist_all
            )
            ratio_4b_err_fvt = (
                4 * np.sqrt(var_ratio_4b_fvt * l2a / hist_all) + 4 * l2a / hist_all
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_lb_count = np.clip(
                ratio_4b_mean_count.reshape(-1) - ratio_4b_err_count, 0, 1
            )
            ratio_ub_count = np.clip(
                ratio_4b_mean_count.reshape(-1) + ratio_4b_err_count, 0, 1
            )
        ax[1].plot(bins_range[:-1], ratio_4b_mean_count, label="Ratio 4b")
        ax[1].fill_between(bins_range[:-1], ratio_lb_count, ratio_ub_count, alpha=0.3)
        ax[1].legend()
        ax[1].set_xlabel("cluster")

        twin_ax0 = ax[0].twinx()
        twin_ax0.plot(bins_range[:-1], ratio_4b_mean_count, label="Ratio 4b")
        ax[0].grid(color="k", linestyle="--", linewidth=0.5, axis="both")

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_lb_fvt = np.clip(
                ratio_4b_mean_fvt.reshape(-1) - ratio_4b_err_fvt, 0, 1
            )
            ratio_ub_fvt = np.clip(
                ratio_4b_mean_fvt.reshape(-1) + ratio_4b_err_fvt, 0, 1
            )
        ax[2].plot(bins_range[:-1], ratio_4b_mean_fvt, label="Ratio 4b")
        ax[2].fill_between(bins_range[:-1], ratio_lb_fvt, ratio_ub_fvt, alpha=0.3)
        ax[2].legend()
        ax[2].set_xlabel("cluster")

        if show_plot:
            plt.show()
            plt.close()
        else:
            return fig
    else:
        fig = go.Figure()
        fig.update_layout(width=600, height=600)
        fig.update_layout(hovermode=False)
        fig.update_layout(
            title="<br>".join(
                [
                    title,
                    "3b: {}, bg 4b: {}, Signal: {}".format(
                        np.sum(is_3b), np.sum(is_bg4b), np.sum(is_signal)
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
                    x=q_repr[is_signal, 0],
                    y=q_repr[is_signal, 1],
                    mode="markers",
                    name="Signal",
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
                    x=q_repr[is_signal, 0],
                    y=q_repr[is_signal, 1],
                    z=q_repr[is_signal, 2],
                    mode="markers",
                    name="Signal",
                    marker=dict(size=3, color="green"),
                )
            )

        fig.show()


def plot_cluster_1d(ax0, ax1, q_repr, is_3b, is_bg4b, is_signal, weights):
    q_repr = q_repr.reshape(-1)
    bins_range = np.linspace(np.min(q_repr), np.max(q_repr), 50)
    hist_3b, _, _ = ax0.hist(
        q_repr[is_3b],
        weights=weights[is_3b],
        bins=bins_range,
        label="bg 3b",
        linewidth=1,
        histtype="step",
        density=False,
    )
    hist_bg4b, _, _ = ax0.hist(
        q_repr[is_bg4b],
        weights=weights[is_bg4b],
        bins=bins_range,
        label="bg 4b",
        linewidth=1,
        histtype="step",
        density=False,
    )
    hist_signal, _, _ = ax0.hist(
        q_repr[is_signal],
        weights=weights[is_signal],
        bins=bins_range,
        label="Signal",
        linewidth=1,
        histtype="step",
        density=False,
    )
    ax0.legend()
    ax0.set_xlabel("cluster")
    # 4b / 3b ratio

    hist_4b = hist_bg4b + hist_signal

    # ignore warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_mean = hist_4b / hist_3b
        ratio_std = np.sqrt(hist_4b * (1 / hist_3b) ** 2 + (hist_4b / hist_3b**2) ** 2)
    alpha = 0.05
    z = stats.norm.ppf(1 - alpha / 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_lb = ratio_mean.reshape(-1) - z * ratio_std.reshape(-1)
        ratio_ub = ratio_mean.reshape(-1) + z * ratio_std.reshape(-1)
    ax1.plot(bins_range[:-1], ratio_mean, label="4b / 3b")
    ax1.fill_between(bins_range[:-1], ratio_lb, ratio_ub, alpha=0.3)
    ax1.legend()
    ax1.set_xlabel("cluster")


def hist_events_by_labels(
    events: EventsData, values: np.ndarray, bins, ax, **hist_kwargs
):
    assert len(values) == len(events)
    ax.hist(
        values[events.is_3b],
        bins=bins,
        histtype="step",
        label="3b",
        weights=events.weights[events.is_3b],
        **hist_kwargs,
    )
    ax.hist(
        values[events.is_bg4b],
        bins=bins,
        histtype="step",
        label="bg4b",
        weights=events.weights[events.is_bg4b],
        **hist_kwargs,
    )
    ax.hist(
        values[events.is_signal],
        bins=bins,
        histtype="step",
        label="signal",
        weights=events.weights[events.is_signal],
        **hist_kwargs,
    )


def plot_sr_stats(events: EventsData, sr_stats: np.ndarray, ax, label, **plot_kwargs):
    assert len(events) == len(sr_stats)

    sr_stats_argsort = np.argsort(sr_stats)[::-1]
    weights = events.weights[sr_stats_argsort]
    is_signal = events.is_signal[sr_stats_argsort]
    is_4b = events.is_4b[sr_stats_argsort]

    ax.plot(
        np.cumsum(weights * is_4b) / np.sum(weights * is_4b),
        np.cumsum(weights * is_signal) / np.sum(weights * is_signal),
        label=label,
        **plot_kwargs,
    )


def plot_reweighted_samples(
    events: EventsData,
    hist_values: np.ndarray,
    reweights: np.ndarray,
    ax: plt.Axes,
    **plot_kwargs,
):
    assert len(events) == len(hist_values) == len(reweights)
    assert ax is not None

    bins = plot_kwargs.get("bins", 20)
    mode = plot_kwargs.get("mode", "uniform")

    x_min, x_max = np.min(hist_values), np.max(hist_values)

    if isinstance(bins, int):
        if mode == "uniform":
            bins = np.linspace(x_min, x_max, bins)
        elif mode == "quantile":
            bins = np.quantile(hist_values, np.linspace(0, 1, bins))
        else:
            raise ValueError(f"Invalid mode: {mode}")

    rw = reweights * events.weights
    rw_sq = reweights**2 * events.weights
    hist_3b, _ = np.histogram(
        hist_values[events.is_3b],
        bins=bins,
        weights=rw[events.is_3b],
    )
    hist_3b_sq, _ = np.histogram(
        hist_values[events.is_3b],
        bins=bins,
        weights=rw_sq[events.is_3b],
    )
    hist_4b, _ = np.histogram(
        hist_values[events.is_4b],
        bins=bins,
        weights=rw[events.is_4b],
    )
    hist_4b_sq, _ = np.histogram(
        hist_values[events.is_4b],
        bins=bins,
        weights=rw_sq[events.is_4b],
    )

    midpoints = (bins[:-1] + bins[1:]) / 2
    ax.stairs(
        hist_3b,
        bins,
        label="reweighted 3b",
        color=plt.get_cmap("tab10").colors[0],
    )
    ax.stairs(hist_4b, bins, label="4b", color=plt.get_cmap("tab10").colors[1])
    ax.errorbar(
        midpoints,
        hist_4b,
        yerr=np.sqrt(hist_3b_sq + hist_4b_sq),
        color=plt.get_cmap("tab10").colors[1],
        capsize=3,
        fmt="o",
        markersize=2,
    )
    ax.legend()
    ax.set_xlabel(plot_kwargs.get("xlabel", ""))

    mask = hist_3b_sq + hist_4b_sq > 0
    sigma = np.full_like(hist_3b, np.nan)
    sigma[mask] = (hist_4b[mask] - hist_3b[mask]) / np.sqrt(
        hist_3b_sq[mask] + hist_4b_sq[mask]
    )
    twin_ax = ax.twinx()
    twin_ax.plot(midpoints, sigma, "o", color="red", markersize=3)
    twin_ax.axhline(0, color="black", linestyle="--")
    twin_ax.set_ylim(-4, 4)
    twin_ax.set_ylabel("sigma")
    # grid
    twin_ax.grid(color="k", linestyle="--", linewidth=0.2, axis="both")


def plot_rewighted_samples_by_model(
    pl_module: pl.LightningModule, events: EventsData, **plot_kwargs
):
    pl_module.eval()
    pl_module.to("cuda")
    fvt_scores = pl_module.predict(events.X_torch).detach().cpu().numpy()[:, 1]
    ratio_4b = plot_kwargs.get("ratio_4b", 0.5)
    reweights = (fvt_scores / (1 - fvt_scores)) * ratio_4b / (1 - ratio_4b)
    reweights = np.where(events.is_3b, reweights, 1)
    fig, ax = plt.subplots(1, 1, figsize=plot_kwargs.get("figsize", (8, 6)))
    fig.suptitle(plot_kwargs.get("title", ""))
    bins = plot_kwargs.get("bins", 30)
    plot_reweighted_samples(
        events,
        fvt_scores,
        reweights,
        ax=ax,
        bins=bins,
        mode="uniform",
    )
    plt.show()
    plt.close("all")


def plot_rewighted_samples_by_model_v2(
    pl_module: pl.LightningModule, events: EventsData, **plot_kwargs
):
    pl_module.eval()
    pl_module.to("cuda")
    fvt_scores = pl_module.predict(events.X_torch).detach().cpu().numpy()[:, 1]
    ratio_4b = plot_kwargs.get("ratio_4b", 0.5)
    reweights = np.where(
        events.is_3b, fvt_scores * ratio_4b, (1 - fvt_scores) * (1 - ratio_4b)
    )
    fig, ax = plt.subplots(1, 1, figsize=plot_kwargs.get("figsize", (8, 6)))
    fig.suptitle(plot_kwargs.get("title", ""))
    bins = plot_kwargs.get("bins", 30)
    plot_reweighted_samples(
        events,
        fvt_scores,
        reweights,
        ax=ax,
        bins=bins,
        mode="uniform",
    )
    plt.show()
    plt.close("all")
