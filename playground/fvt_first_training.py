import torch
import pathlib
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import click


from events_data import EventsData
from fvt_classifier import FvTClassifier


@click.command()
@click.option("--signal_ratio", type=float)
@click.option("--four_tag_ratio", type=float, default=0.5)
@click.option("--dim_dijet_features", type=int, default=6)
@click.option("--dim_quadjet_features", type=int, default=6)
@click.option("--n_3b", type=int, default=250000)
@click.option("--n_all4b", type=int, default=250000)
@click.option("--test_ratio", type=float, default=0.5)
@click.option("--val_ratio", type=float, default=0.33)
@click.option("--batch_size", type=int, default=1024)
@click.option("--max_epochs", type=int, default=50)
@click.option("--seed", type=int)
def main(
    signal_ratio,
    four_tag_ratio,
    dim_dijet_features,
    dim_quadjet_features,
    n_3b,
    n_all4b,
    test_ratio,
    val_ratio,
    batch_size,
    max_epochs,
    seed,
):

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

    print("3b-jet events: ", len(df_3b))
    print("4b-jet events: ", len(df_bg4b))
    print("HH4b-jet events: ", len(df_signal))

    # Training

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

    ###########################################################################################
    ###########################################################################################
    pl.seed_everything(seed)
    np.random.seed(seed)

    events_3b = EventsData.from_dataframe(df_3b, features)
    events_3b.shuffle(seed=seed)

    events_bg4b = EventsData.from_dataframe(df_bg4b, features)
    events_bg4b.shuffle(seed=seed)
    events_bg4b.trim(n_all4b - int(n_all4b * signal_ratio))

    events_signal = EventsData.from_dataframe(df_signal, features)
    events_signal.shuffle(seed=seed)
    events_signal.trim(int(n_all4b * signal_ratio))

    assert signal_ratio == 0 or int(n_all4b * signal_ratio) > 0

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

    # reduce number of 4b samples to 1/8
    print(
        "4b ratio: ",
        events_train.total_weight_4b / events_train.total_weight,
    )
    print(
        "Signal ratio: ",
        events_train.total_weight_signal / events_train.total_weight_4b,
    )

    events_train.shuffle(seed=seed)
    events_train, _ = events_train.split(
        1 - test_ratio, name_1="fvt_train", name_2="fvt_test", seed=seed
    )
    events_train.shuffle(seed=seed)
    events_train, events_val = events_train.split(
        1 - val_ratio, name_1="fvt_train", name_2="fvt_val", seed=seed
    )
    events_train.fit_batch_size(batch_size)
    events_val.fit_batch_size(batch_size)

    ###########################################################################################
    ###########################################################################################

    num_classes = 2
    dim_input_jet_features = 4
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
    lr = 1e-3

    pl.seed_everything(seed)

    model = FvTClassifier(
        num_classes,
        dim_input_jet_features,
        dim_dijet_features,
        dim_quadjet_features,
        run_name=run_name,
        device=torch.device("cuda:0"),
        lr=lr,
    )

    model.fit(
        events_train.to_tensor_dataset(),
        events_val.to_tensor_dataset(),
        batch_size=batch_size,
        max_epochs=max_epochs,
    )


if __name__ == "__main__":
    main()
