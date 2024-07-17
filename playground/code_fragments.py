import pathlib
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt

from events_data import EventsData


def routine_1(
    signal_ratio: float,
    seed: int,
    four_tag_ratio: float = 0.5,
    dim_dijet_features: int = 6,
    dim_quadjet_features: int = 6,
    n_3b: int = 250000,
    n_all4b: int = 250000,
    test_ratio: float = 0.5,
    val_ratio: float = 0.33,
    batch_size: int = 1024,
    max_epochs: int = 50,
    return_keys: list[str] = ["train", "val", "test"],
) -> dict[str, EventsData]:

    directory = pathlib.Path("../events/MG3")

    df_3b = pd.read_hdf(directory / "dataframes" / "threeTag_picoAOD.h5")
    df_3b = df_3b.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_3b = df_3b.iloc[:n_3b]

    df_bg4b = pd.read_hdf(directory / "dataframes" / "fourTag_10x_picoAOD.h5")
    df_bg4b = df_bg4b.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_bg4b = df_bg4b.iloc[:n_all4b]

    df_signal = pd.read_hdf(directory / "dataframes" / "HH4b_picoAOD.h5")
    df_signal = df_signal.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_signal = df_signal.iloc[:n_all4b]

    df_3b["signal"] = False
    df_bg4b["signal"] = False
    df_signal["signal"] = True

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

    pl.seed_everything(seed)
    np.random.seed(seed)

    events_3b = EventsData.from_dataframe(df_3b, features)
    events_3b.shuffle(seed=seed)

    events_bg4b = EventsData.from_dataframe(df_bg4b, features)
    events_bg4b.shuffle(seed=seed)
    events_bg4b.trim(n_all4b - int(n_all4b * signal_ratio))

    assert signal_ratio == 0 or int(n_all4b * signal_ratio) > 0

    events_signal = EventsData.from_dataframe(df_signal, features)
    events_signal.shuffle(seed=seed)
    events_signal.trim(int(n_all4b * signal_ratio))

    # set weight ratio to be exactly signal ratio
    if len(events_signal) > 0:
        new_hh4b_weights = (
            (signal_ratio / (1 - signal_ratio))
            * (events_bg4b.total_weight / events_signal.total_weight)
            * events_signal.weights
        )
        events_signal.reweight(new_hh4b_weights)

    # set four tag ratio to be exactly four_tag_ratio

    new_3b_weights = (
        (four_tag_ratio / (1 - four_tag_ratio))
        * (
            (events_bg4b.total_weight + events_signal.total_weight)
            / events_3b.total_weight
        )
        * events_3b.weights
    )
    events_3b.reweight(new_3b_weights)

    events_train = EventsData.merge([events_3b, events_bg4b, events_signal])

    events_train.shuffle(seed=seed)
    events_train, events_test = events_train.split(
        1 - test_ratio, name_1="fvt_train", name_2="fvt_test", seed=seed
    )

    events_train.shuffle(seed=seed)
    events_train, events_val = events_train.split(
        0.67, name_1="fvt_train", name_2="fvt_val", seed=seed
    )

    events_train.fit_batch_size(batch_size)
    events_val.fit_batch_size(batch_size)

    from fvt_classifier import FvTClassifier

    run_name = "_".join(
        [
            "fvt_picoAOD",
            f"signal_ratio={signal_ratio}",
            f"four_tag_ratio={four_tag_ratio}",
            f"dijet={dim_dijet_features}",
            f"quadjet={dim_quadjet_features}",
            f"n_3b={n_3b}",
            f"n_all4b={n_all4b}",
            f"epochs={max_epochs}",
            f"batch_size={batch_size}",
            f"val_ratio={val_ratio}",
            f"test_ratio={test_ratio}",
            f"seed={seed}",
        ]
    )

    fvt_model = FvTClassifier.load_from_checkpoint(
        f"./checkpoints/{run_name}_best.ckpt"
    )
    fvt_model.eval()
    device = torch.device("cuda:0")
    fvt_model = fvt_model.to(device)

    assert all([k in ["train", "val", "test"] for k in return_keys])

    return_dict = {}

    if "train" in return_keys:
        events_train.set_model_scores(fvt_model)
        return_dict["train"] = events_train

    if "val" in return_keys:
        events_val.set_model_scores(fvt_model)
        return_dict["val"] = events_val

    if "test" in return_keys:
        events_test.set_model_scores(fvt_model)
        return_dict["test"] = events_test

    # logits = fvt_model(events_train.X_torch.to(device))
    # loss = fvt_model.loss(logits, events_train.is_4b_torch.to(device)).detach().cpu()
    # train_loss = (
    #     (events_train.weights_torch * loss).sum() / events_train.weights_torch.sum()
    # ).numpy()

    # logits = fvt_model(events_val.X_torch.to(device))
    # loss = fvt_model.loss(logits, events_val.is_4b_torch.to(device)).detach().cpu()
    # val_loss = (
    #     (events_val.weights_torch * loss).sum() / events_val.weights_torch.sum()
    # ).numpy()

    # logits = fvt_model(events_test.X_torch.to(device))
    # loss = fvt_model.loss(logits, events_test.is_4b_torch.to(device)).detach().cpu()
    # test_loss = (
    #     (events_test.weights_torch * loss).sum() / events_test.weights_torch.sum()
    # ).numpy()

    return return_dict


def fvt_score_hist(events_data: EventsData, ax: plt.Axes = None):
    is_3b, is_bg4b, is_signal = (
        events_data.is_3b,
        events_data.is_bg4b,
        events_data.is_signal,
    )
    is_4b, w = events_data.is_4b, events_data.weights
    fvt_score = events_data.fvt_score
    att_q_repr = events_data.att_q_repr

    bins_range = np.linspace(0, 1, 50)
    hist_3b, _, _ = ax.hist(
        fvt_score[is_3b],
        bins=bins_range,
        label="bg 3b",
        linewidth=1,
        histtype="step",
        density=False,
        weights=w[is_3b],
    )
    hist_bg4b, _, _ = ax.hist(
        fvt_score[is_bg4b],
        bins=bins_range,
        label="bg 4b",
        linewidth=1,
        histtype="step",
        density=False,
        weights=w[is_bg4b],
    )
    hist_signal, _, _ = ax.hist(
        fvt_score[is_signal],
        bins=bins_range,
        label="HH 4b",
        linewidth=1,
        histtype="step",
        density=False,
        weights=w[is_signal],
    )
    ax.legend()
    ax.set_xlabel("FvT output")


def att_q_repr_hist(events_data: EventsData, dim_quadjet_features: int):
    is_3b, is_bg4b, is_signal = (
        events_data.is_3b,
        events_data.is_bg4b,
        events_data.is_signal,
    )
    fig, ax = plt.subplots(nrows=dim_quadjet_features, ncols=2, figsize=(12, 24))
    w = events_data.weights
    att_q_repr = events_data.att_q_repr

    for i in range(dim_quadjet_features):
        ax[i, 0].hist(
            att_q_repr[is_3b, i],
            bins=50,
            label="bg 3b",
            linewidth=1,
            histtype="step",
            density=False,
            weights=w[is_3b],
        )
        ax[i, 0].hist(
            att_q_repr[is_bg4b, i],
            bins=50,
            label="bg 4b",
            linewidth=1,
            histtype="step",
            density=False,
            weights=w[is_bg4b],
        )
        ax[i, 0].hist(
            att_q_repr[is_signal, i],
            bins=50,
            label="HH 4b",
            linewidth=1,
            histtype="step",
            density=False,
            weights=w[is_signal],
        )
        ax[i, 0].legend()
        ax[i, 0].set_xlabel(f"Feature {i}")

        ax[i, 1].hist(
            att_q_repr[is_3b, i],
            bins=50,
            label="bg 3b",
            linewidth=1,
            histtype="step",
            density=True,
            weights=w[is_3b],
        )
        ax[i, 1].hist(
            att_q_repr[is_bg4b, i],
            bins=50,
            label="bg 4b",
            linewidth=1,
            histtype="step",
            density=True,
            weights=w[is_bg4b],
        )
        ax[i, 1].hist(
            att_q_repr[is_signal, i],
            bins=50,
            label="HH 4b",
            linewidth=1,
            histtype="step",
            density=True,
            weights=w[is_signal],
        )
        ax[i, 1].legend()
        ax[i, 1].set_xlabel(f"Feature {i}")

    plt.show()
    plt.close()
