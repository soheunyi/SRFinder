from datetime import datetime

print("Let's start!, current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
import matplotlib.pyplot as plt
import numpy as np
import torch
from signal_region import get_SR_CR_cut
from events_data import events_from_scdinfo
from training_info import TrainingInfo
from dataset import MotherSamples
from plots import plot_rewighted_samples_by_model
import pathlib
import pandas as pd

print("Packages loaded, current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

features = [
    "sym_Jet0_pt",
    "sym_Jet1_pt",
    "sym_Jet2_pt",
    "sym_Jet3_pt",
    "sym_Jet0_eta",
    "sym_Jet1_eta",
    "sym_Jet2_eta",
    "sym_Jet3_eta",
    "sym_Jet0_phi",
    "sym_Jet1_phi",
    "sym_Jet2_phi",
    "sym_Jet3_phi",
    "sym_Jet0_m",
    "sym_Jet1_m",
    "sym_Jet2_m",
    "sym_Jet3_m",
]

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["figure.titlesize"] = 25
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 20
# increase tick label size
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15

n_3b = 100_0000
device = torch.device("cuda")
experiment_name = "CR_fvt_training_v2"
signal_filename = "HH4b_picoAOD.h5"
ratio_4b = 0.5
nbins = 10
signal_ratio = 0.02
seed = 1


hparam_filter = {
    "experiment_name": experiment_name,
    "dataset": lambda x: all(
        [x["seed"] == seed, x["n_3b"] == n_3b, x["signal_ratio"] == signal_ratio]
    ),
    "aux_info_step": 3,
    "model": "FvTClassifier",
}
hashes = TrainingInfo.find(hparam_filter)
assert len(hashes) == 1

print(
    "Loading training info, current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)

CR_fvt_tinfo = TrainingInfo.load(hashes[0])
smeared_fvt_hash = CR_fvt_tinfo.hparams["smeared_fvt_hash"]
base_encoder_hash = CR_fvt_tinfo.hparams["encoder_hash"]

base_fvt_model = TrainingInfo.load(base_encoder_hash).load_trained_model("best")
base_fvt_model.eval()
base_fvt_model.to(torch.device("cuda"))

smeared_fvt_tinfo = TrainingInfo.load(smeared_fvt_hash)
smeared_fvt_model = smeared_fvt_tinfo.load_trained_model("best")
smeared_fvt_model.eval()
smeared_fvt_model.to(torch.device("cuda"))

# Use the same mother samples and exclude ones used for training base & smeared FvT model
msamples = MotherSamples.load(smeared_fvt_tinfo.ms_hash)
tst_scdinfo = msamples.scdinfo[~smeared_fvt_tinfo.ms_idx]
SR_stats_train = smeared_fvt_tinfo.aux_info["SR_stats_train"]
SR_stats = smeared_fvt_tinfo.aux_info["SR_stats_tst"]
events_tst = events_from_scdinfo(tst_scdinfo, features, signal_filename)
events_train = events_from_scdinfo(
    msamples.scdinfo[smeared_fvt_tinfo.ms_idx], features, signal_filename
)
SR_cut, CR_cut = get_SR_CR_cut(
    SR_stats_train, events_train, CR_fvt_tinfo.hparams["signal_region"]
)

SR_idx = SR_stats >= SR_cut
CR_idx = (SR_stats >= CR_cut) & (SR_stats < SR_cut)

events_tst_SR = events_tst[SR_idx]
events_tst_CR = events_tst[CR_idx]
SR_stats_SR = SR_stats[SR_idx]
SR_stats_SR = np.exp(SR_stats_SR)
SR_stats_CR = SR_stats[CR_idx]
SR_stats_CR = np.exp(SR_stats_CR)

events_train_SR = events_train[SR_stats_train >= SR_cut]
SR_stats_train_SR = SR_stats_train[SR_stats_train >= SR_cut]

CR_fvt_model = CR_fvt_tinfo.load_trained_model("best")
CR_fvt_model.eval()
CR_fvt_model.to(torch.device("cuda"))

print(
    "Computing FvT scores, current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)

fvt_scores_SR = CR_fvt_model.predict(events_tst_SR.X_torch)[:, 1].detach().cpu().numpy()
fvt_scores_CR = CR_fvt_model.predict(events_tst_CR.X_torch)[:, 1].detach().cpu().numpy()


print("Plotting, current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

for hist_values_SR, hist_values_CR, name in zip(
    [SR_stats_SR],
    [SR_stats_CR],
    ["SR_stats"],
):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    if name == "SR_stats":
        label = r"$\xi = \gamma / \widetilde{\gamma}$"
    else:
        label = r"$p_{4b} / (p_{3b} + p_{4b})$"
    fig.supxlabel(label)

    if name == "SR_stats":
        SR_bins = np.linspace(np.min(hist_values_SR), np.max(hist_values_SR), nbins + 1)
    else:
        SR_bins = np.linspace(0, 1, 33)

    reweights_SR = fvt_scores_SR / (1 - fvt_scores_SR)
    reweights_SR = np.where(events_tst_SR.is_4b, 1, -reweights_SR)
    plot_rewighted_samples_by_model(
        events_tst_SR,
        hist_values_SR,
        fvt_scores_SR,
        ax=ax,
        bins=SR_bins,
        disable_twin_ax=True,
        errorbar="3b",
        no_4b=True,
    )
    hist_bg4b, _ = np.histogram(
        hist_values_SR[events_tst_SR.is_bg4b],
        bins=SR_bins,
        weights=events_tst_SR.weights[events_tst_SR.is_bg4b],
    )
    hist_signal, _ = np.histogram(
        hist_values_SR[events_tst_SR.is_signal],
        bins=SR_bins,
        weights=events_tst_SR.weights[events_tst_SR.is_signal],
    )
    hist_3b, _ = np.histogram(
        hist_values_SR[events_tst_SR.is_3b],
        bins=SR_bins,
        weights=events_tst_SR.weights[events_tst_SR.is_3b],
    )
    ax.stairs(
        hist_bg4b, SR_bins, label="Background 4b", color=plt.cm.tab10(1), linestyle="--"
    )
    ax.stairs(hist_signal, SR_bins, label="Signal", color=plt.cm.tab10(2))
    ax.stairs(hist_3b, SR_bins, label="3b", color=plt.cm.tab10(0), linestyle="--")
    ax.legend()
    ax.set_title("Signal Region")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Counts")

    plt.tight_layout()
    plt.savefig(f"./figures/background_4b_estimation_{name}.pdf", dpi=300)
    plt.show()
    plt.close("all")


from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

x_range = np.linspace(-1, 1, 1000)
pdf_1 = stats.norm.pdf(x_range, loc=0, scale=1)

signal_center = 0.25
signal_width = 0.1
pdf_2 = 0.05 * stats.norm.pdf(x_range, loc=signal_center, scale=signal_width)

sr_x1 = signal_center - 3 * signal_width
sr_x2 = signal_center + 3 * signal_width

in_SR = (x_range >= sr_x1) & (x_range <= sr_x2)
in_CR_left = x_range < sr_x1
in_CR_right = x_range > sr_x2
in_CR = in_CR_left | in_CR_right

fig, axs = plt.subplots(3, 1, figsize=(12, 20))
# fig.supxlabel("$x$", fontsize=30)
fig.supylabel("Density", fontsize=30)

lw = 6

axs[0].plot(
    x_range[in_CR_left],
    pdf_1[in_CR_left],
    color=plt.cm.tab10.colors[1],
    label="Estimated Background (CR)",
    linewidth=lw,
)
axs[0].plot(
    x_range[in_CR_right], pdf_1[in_CR_right], color=plt.cm.tab10.colors[1], linewidth=lw
)

axs[1].plot(
    x_range[in_CR_left], pdf_1[in_CR_left], color=plt.cm.tab10.colors[1], linewidth=lw
)
axs[1].plot(
    x_range[in_CR_right], pdf_1[in_CR_right], color=plt.cm.tab10.colors[1], linewidth=lw
)
axs[1].plot(
    x_range[in_SR],
    pdf_1[in_SR],
    linestyle="--",
    label="Interpolated Background (SR)",
    color=plt.cm.tab10.colors[1],
    linewidth=lw,
)

axs[2].plot(
    x_range[in_CR_left], pdf_1[in_CR_left], color=plt.cm.tab10.colors[1], linewidth=lw
)
axs[2].plot(
    x_range[in_CR_right], pdf_1[in_CR_right], color=plt.cm.tab10.colors[1], linewidth=lw
)
axs[2].plot(
    x_range[in_SR],
    pdf_1[in_SR],
    linestyle="--",
    label="Interpolated Background (SR)",
    color=plt.cm.tab10.colors[1],
    linewidth=lw,
)
axs[2].plot(
    x_range[in_SR],
    pdf_1[in_SR] + pdf_2[in_SR],
    color=plt.cm.tab10.colors[2],
    label="Signal + Background",
    linewidth=lw,
)


CR_color = "#0070c0"
SR_color = "#4ea230"

for ax in axs:
    ax.axvline(signal_center, color="black", linestyle="--", linewidth=lw)
    ax.fill_betweenx(
        y=[0, 1], x1=sr_x1, x2=sr_x2, color=SR_color, alpha=0.3, label="SR"
    )
    ax.fill_between(
        x_range, y1=0, y2=1, where=in_CR, color=CR_color, alpha=0.3, label="CR"
    )
    ax.set_ylim(0.1, 0.7)
    ax.set_xlim(-1, 1)
    ax.legend(loc="upper left", fontsize=25)

plt.tight_layout()
plt.savefig("./figures/bg_estimation_toy_example.png", dpi=300)
plt.show()
plt.close()

# density plot
from matplotlib import patches as mpatches

directory = pathlib.Path("../events/MG3")

df_3b = pd.read_hdf(directory / "dataframes" / "threeTag_picoAOD.h5")
df_bg4b = pd.read_hdf(directory / "dataframes" / "fourTag_10x_picoAOD.h5")
df_hh4b = pd.read_hdf(directory / "dataframes" / "HH4b_picoAOD.h5")

df_3b["signal"] = False
df_bg4b["signal"] = False
df_hh4b["signal"] = True

SR_fn = lambda dm_0, dm_1: np.sqrt((1 - m_H / dm_0) ** 2 + (1 - m_H / dm_1) ** 2)
CR_fn = lambda dm_0, dm_1: np.sqrt(
    (dm_0 - sigma_C * m_H) ** 2 + (dm_1 - sigma_C * m_H) ** 2
)


CR_color = "#0070c0"
SR_color = "#4ea230"


# use latex
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["figure.labelsize"] = 15

m_H = 125
sigma_C = 1
K_s = 1
K_c = 1
sigma_C = 1.03
K_s = 0.16
K_c = 30

SR_fn = lambda dm_0, dm_1: np.sqrt((1 - m_H / dm_0) ** 2 + (1 - m_H / dm_1) ** 2)
CR_fn = lambda dm_0, dm_1: np.sqrt(
    (dm_0 - sigma_C * m_H) ** 2 + (dm_1 - sigma_C * m_H) ** 2
)

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
fig.supxlabel("$m_1 (\mathrm{GeV})$")
fig.supylabel("$m_2 (\mathrm{GeV})$")
for ax, (name, df) in zip(
    axs,
    [
        ("3b", df_3b),
        (r"Background $\mathrm{4b}$", df_bg4b),
        (r"$\mathrm{HH} \rightarrow 4b$", df_hh4b),
    ],
):
    m_diff = np.stack(
        [
            df["m01"] - df["m23"],
            df["m02"] - df["m13"],
            df["m03"] - df["m12"],
        ],
        axis=1,
    )
    m_diff = np.abs(m_diff)
    m_diff_argmin = np.argmin(m_diff, axis=1)
    m1 = np.where(
        m_diff_argmin == 0,
        df["m01"],
        np.where(m_diff_argmin == 1, df["m02"], df["m03"]),
    )
    m2 = np.where(
        m_diff_argmin == 0,
        df["m23"],
        np.where(m_diff_argmin == 1, df["m13"], df["m12"]),
    )

    print(f"{name} in SR: {np.mean(SR_fn(m1, m2) < K_s)}")

    # heatmap
    m_min = 75
    m_max = 175
    xedges = np.linspace(m_min, m_max, 30)
    yedges = np.linspace(m_min, m_max, 30)
    H, xedges, yedges = np.histogram2d(m1, m2, bins=(xedges, yedges))
    H = H.T
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H, shading="auto", cmap="magma")

    # add colorbar
    cbar = plt.colorbar(
        ax.pcolormesh(X, Y, H, shading="auto", cmap="magma"), fraction=0.047, pad=0.01
    )
    cbar.set_label("Counts")

    ax.set_xlim(m_min, m_max)
    ax.set_ylim(m_min, m_max)

    ax.contour(X, Y, SR_fn(X, Y), levels=[K_s], colors=SR_color, linewidths=4)
    ax.contour(X, Y, CR_fn(X, Y), levels=[K_c], colors=CR_color, linewidths=4)

    # add manual legend

    SR_patch = mpatches.Patch(color=SR_color, label="SR")
    CR_patch = mpatches.Patch(color=CR_color, label="CR")
    ax.legend(handles=[SR_patch, CR_patch], loc="lower left")
    ax.set_aspect("equal")
    ax.set_title(name)

plt.tight_layout()
plt.savefig("./figures/m1m2_heatmap.png", dpi=300)
plt.show()
plt.close()


# density plot
import numpy as np
from matplotlib import patches as mpatches

SR_color = "#4ea230"
CR_color = "#0070c0"

# use latex
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["figure.labelsize"] = 15
plt.rcParams["figure.titlesize"] = 15

m_H = 125
sigma_C = 1
K_s = 1
K_c = 1
sigma_C = 1.03
K_s = 0.16
K_c = 30

import matplotlib.pyplot as plt
import pathlib
import pandas as pd


m_diff = np.stack(
    [
        df_bg4b["m01"] - df_bg4b["m23"],
        df_bg4b["m02"] - df_bg4b["m13"],
        df_bg4b["m03"] - df_bg4b["m12"],
    ],
    axis=1,
)
m_diff = np.abs(m_diff)
m_diff_argmin = np.argmin(m_diff, axis=1)
m1 = np.where(
    m_diff_argmin == 0,
    df_bg4b["m01"],
    np.where(m_diff_argmin == 1, df_bg4b["m02"], df_bg4b["m03"]),
)
m2 = np.where(
    m_diff_argmin == 0,
    df_bg4b["m23"],
    np.where(m_diff_argmin == 1, df_bg4b["m13"], df_bg4b["m12"]),
)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

# heatmap
m_min = 75
m_max = 175
xedges = np.linspace(m_min, m_max, 30)
yedges = np.linspace(m_min, m_max, 30)
H, xedges, yedges = np.histogram2d(m1, m2, bins=(xedges, yedges))
H = H.T
X, Y = np.meshgrid(xedges, yedges)


for ax in axs:
    ax.pcolormesh(X, Y, H, shading="auto", cmap="magma")
    ax.set_xlim(m_min, m_max)
    ax.set_ylim(m_min, m_max)
    ax.set_aspect("equal")
    ax.contour(X, Y, SR_fn(X, Y), levels=[K_s], colors=SR_color, linewidths=4)
    ax.contour(X, Y, CR_fn(X, Y), levels=[K_c], colors=CR_color, linewidths=4)


axs[0].contourf(X, Y, SR_fn(X, Y), levels=[0, K_s], colors="white")
axs[0].contourf(X, Y, CR_fn(X, Y), levels=[0, np.inf], colors="white")
axs[1].contourf(X, Y, SR_fn(X, Y), levels=[0, K_s], colors="white")
axs[1].contourf(X, Y, CR_fn(X, Y), levels=[K_c, np.inf], colors="white")
# axs[2].contourf(X, Y, SR_fn(X, Y), levels=[0, K_s], colors="white")
axs[2].contourf(X, Y, CR_fn(X, Y), levels=[K_c, np.inf], colors="white")

for ax in axs:
    SR_patch = mpatches.Patch(color=SR_color, label="SR")
    CR_patch = mpatches.Patch(color=CR_color, label="CR")
    ax.legend(handles=[SR_patch, CR_patch], loc="lower left")

plt.tight_layout()
plt.savefig("./figures/SR_bg4b_interpolation.png", dpi=300)
plt.show()
plt.close()
