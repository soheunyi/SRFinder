import pathlib
import click
import torch
from autoencoder import Autoencoder
import pytorch_lightning as pl
import yaml
import pandas as pd

import sys

sys.path.append("/home/soheuny/HH4bsim")

from python.classifier.symmetrize_df import symmetrize_df
from python.classifier.symmetrized_model_train import symmetrizedModelParameters
from fvt_eval import calculate_fvt_values
from split_samples import AE_3B_INDEX, AE_4B_INDEX


@click.command()
@click.option("--config", help="Path to the config file")
def main(config):
    # parse yaml
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    pl.seed_everything(config["seed"])
    # load data
    data_directory = pathlib.Path("/home/soheuny/HH4bsim/events/MG3")

    df3B = pd.read_hdf(data_directory / "dataframes" / "symmetrized_bbbj.h5")
    df4B = pd.read_hdf(data_directory / "dataframes" / "symmetrized_bbbb_large.h5")

    df = pd.concat([df3B.iloc[AE_3B_INDEX], df4B.iloc[AE_4B_INDEX]], sort=False)
    df["mcPseudoTagWeight"] = df["weight"]

    # Check required features
    features = [
        "sym_canJet0_pt",
        "sym_canJet1_pt",
        "sym_canJet2_pt",
        "sym_canJet3_pt",
        "sym_canJet0_eta",
        "sym_canJet1_eta",
        "sym_canJet2_eta",
        "sym_canJet3_eta",
        "sym_canJet1_phi",
        "sym_canJet2_phi",
        "sym_canJet3_phi",
        "sym_canJet0_m",
        "sym_canJet1_m",
        "sym_canJet2_m",
        "sym_canJet3_m",
    ]
    assert all(
        [feature in df.columns for feature in features + ["sym_canJet0_phi"]]
    ), f"Required features not found in the dataframe. Required features: {features + ['sym_canJet0_phi']}"

    X = torch.tensor(df[features].values, dtype=torch.float32)

    sym_model = symmetrizedModelParameters(
        df3B, df4B, fileName=config["sym_model_path"]
    )
    _, grads, scores = calculate_fvt_values(sym_model, df)
    probs_4b_est = torch.softmax(scores, dim=1)[:, 1].detach()
    X = torch.cat((X, probs_4b_est.view(-1, 1)), dim=1)

    if config["use_gradients"]:
        # remove the gradient of sym_canJet0_phi
        grads = grads[:, torch.arange(grads.shape[1]) != 8]
        X = torch.cat((X, grads), dim=1)

    model = Autoencoder(
        input_dim=X.shape[1],
        latent_dim=config["latent_dim"],
        run_name_base=config["run_name_base"],
        hidden_dims=config["hidden_dims"],
        lr=config["lr"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # split X into training and validation
    random_indices = torch.randperm(X.shape[0])

    train_data_ratio = config["train_data_ratio"]
    train_indices = random_indices[: int(train_data_ratio * X.shape[0])]
    validation_indices = random_indices[int(train_data_ratio * X.shape[0]) :]

    X_train = X[train_indices]
    X_validation = X[validation_indices]

    model.set_loss_weights(1 / torch.var(X_train, dim=0))
    model.fit(
        X_train,
        X_validation,
        batch_size=config["batch_size"],
        max_epochs=config["max_epochs"],
    )


if __name__ == "__main__":
    main()
