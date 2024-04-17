import torch
import pathlib
import pandas as pd
import sys
from torch.utils.data import TensorDataset
from variational_autoencoder import VariationalAutoencoder
import pytorch_lightning as pl

directory = pathlib.Path("../events/MG3")

df3b = pd.read_hdf(directory / "dataframes" / "bbbj.h5")
df4b = pd.read_hdf(directory / "dataframes" / "bbbb_large.h5")


sys.path.append("/home/soheuny/HH4bsim/python/classifier/")
from symmetrize_df import symmetrize_df
from symmetrized_model_train import symmetrizedModelParameters

df3b = symmetrize_df(df3b)
df4b = symmetrize_df(df4b)


model_config = "FvT_ResNet_6_6_6_np799_lr0.01_epochs30_stdscale_epoch30_loss0.6703.pkl"

model_filename = (
    f"/home/soheuny/HH4bsim/python/classifier/FvT/fvt_fit/archive/{model_config}"
)
clf = symmetrizedModelParameters(df3b, df4b, fileName=model_filename)

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

X_train = torch.tensor(clf.df_train[features].values, dtype=torch.float32)
X_validation = torch.tensor(clf.df_val[features].values, dtype=torch.float32)

y_train = torch.tensor(clf.df_train["d4"].values, dtype=torch.float32)
y_validation = torch.tensor(clf.df_val["d4"].values, dtype=torch.float32)

weights = clf.df_val[clf.weight].values
w_validation = torch.tensor(weights, dtype=torch.float32)
# add sym_canJet0_phi
X_validation = torch.cat(
    (X_validation[:, :8], torch.zeros_like(X_validation[:, :1]), X_validation[:, 8:]),
    dim=1,
)


pl.seed_everything(42)

# split X_validation into training and validation
random_indices = torch.randperm(len(X_validation))

# For ghostbatch, let len(train_indices) be a multiple of 32
split_at = 32 * (int((2 / 3) * len(X_validation)) // 32)
end_at = 32 * (len(X_validation) // 32)

train_indices = random_indices[:split_at]
validation_indices = random_indices[split_at:end_at]

X_ae_train = X_validation[train_indices]
X_ae_validation = X_validation[validation_indices]
y_ae_train = y_validation[train_indices]
y_ae_validation = y_validation[validation_indices]
w_ae_train = w_validation[train_indices]
w_ae_validation = w_validation[validation_indices]

ae_train_dataset = TensorDataset(X_ae_train, w_ae_train)
ae_validation_dataset = TensorDataset(X_ae_validation, w_ae_validation)


hidden_dims = [256] * 4


likelihood_configs = [
    # pt
    {"type": "TruncatedGaussian", "lb": 0},
    {"type": "TruncatedGaussian", "lb": 0},
    {"type": "TruncatedGaussian", "lb": 0},
    {"type": "TruncatedGaussian", "lb": 0},
    # eta
    {"type": "TruncatedGaussian", "lb": 0, "ub": 2.5},
    {"type": "TruncatedGaussian", "lb": -2.5, "ub": 2.5},
    {"type": "TruncatedGaussian", "lb": -2.5, "ub": 2.5},
    {"type": "TruncatedGaussian", "lb": -2.5, "ub": 2.5},
    # phi
    {"type": "TruncatedGaussian", "lb": 0, "ub": 0.1},  # sym_canJet0_phi should be 0
    {"type": "TruncatedGaussian", "lb": 0, "ub": torch.pi},
    {"type": "TruncatedGaussian", "lb": -torch.pi, "ub": torch.pi},
    {"type": "TruncatedGaussian", "lb": -torch.pi, "ub": torch.pi},
    # m
    {"type": "Gaussian"},
    {"type": "Gaussian"},
    {"type": "Gaussian"},
    {"type": "Gaussian"},
]


for lr in [1e-4, 1e-3]:
    for beta in [0.5, 0.8]:
        for latent_dim in [4, 8]:
            for encoder_type in ["FvTEncoder", "MLPEncoder"]:
                if encoder_type == "MLPEncoder":
                    encoder_config = {
                        "type": "MLPEncoder",
                        "input_dim": X_validation.shape[1],
                        "hidden_dims": hidden_dims[:-1],
                        "latent_dim": hidden_dims[-1],
                        "name": "mlp",
                    }
                elif encoder_type == "FvTEncoder":
                    encoder_config = {
                        "type": "FvTEncoder",
                        "dim_input_jet_features": 4,
                        "dim_intermed_dijet_features": 128,
                        "dim_intermed_quadjet_features": 128,
                        "dim_output": 128,
                        "name": "fvt",
                    }
                else:
                    raise ValueError("Invalid encoder type")
            decoder_config = {
                "type": "MLPDecoder",
                "latent_dim": latent_dim,
                "hidden_dims": hidden_dims[::-1],
                "output_dim": X_validation.shape[1],
            }
            model = VariationalAutoencoder(
                input_dim=X_validation.shape[1],
                latent_dim=latent_dim,
                run_name=f"{encoder_config['name']}_vae_latent_dim={latent_dim}_beta={beta}_lr={lr}",
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                likelihood_configs=likelihood_configs,
                lr=lr,
                beta=beta,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.fit(
                ae_train_dataset,
                ae_validation_dataset,
                batch_size=2**10,
                max_epochs=100,
            )
