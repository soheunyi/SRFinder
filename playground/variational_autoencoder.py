# Variational autoencoder

from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from autoencoder import Encoder, Decoder, Activation
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt


class VariationalAutoencoder(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        run_name: str,
        activation: Literal["ReLU", "RReLU", "LeakyReLU", "SiLU", "ELU"] = "SiLU",
        lr: float = 1e-3,
        loss_weights: torch.Tensor | None = None,
        beta: float = 1.0,
    ):
        super().__init__()
        assert 0 <= beta <= 1
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.run_name = run_name
        self.activation = activation
        self.lr = lr
        self.__loss_weights = loss_weights
        self.beta = beta

        if loss_weights is not None:
            assert loss_weights.shape[0] == input_dim
            self.__loss_weights = loss_weights
        else:
            self.__loss_weights = torch.ones(input_dim)

        self.lr = lr
        self.train_losses = torch.tensor([])
        self.validation_losses = torch.tensor([])
        self.validation_kls = torch.tensor([])
        self.validation_reconstruction_losses = torch.tensor([])
        self.train_batchsizes = torch.tensor([])
        self.validation_batchsizes = torch.tensor([])

        self.best_val_loss = torch.inf
        self.checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/",
            filename=f"{run_name}_best",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )

        self.encoder = Encoder(
            input_dim,
            hidden_dims[-1],
            hidden_dims[:-1],
            activation=activation,
            last_bias=True,
        )
        self.encoder_activation = Activation(activation)
        self.mu_encoder = nn.Linear(hidden_dims[-1], latent_dim, bias=False)
        self.logvar_encoder = nn.Linear(hidden_dims[-1], latent_dim, bias=False)
        self.decoder = Decoder(
            latent_dim, input_dim, hidden_dims[::-1], last_bias=False
        )
        # define scale parameter
        self.log_input_var = nn.Parameter(torch.zeros(input_dim))
        self.latent_mu = nn.Parameter(torch.zeros(latent_dim))
        self.latent_logvar = nn.Parameter(torch.zeros(latent_dim))

    def encode(self, x):
        x = self.encoder(x)
        x = self.encoder_activation(x)
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        mu_x_reconstructed = self.decoder(z)

        return mu_x_reconstructed, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def recons_and_kl(self, x: torch.Tensor, w: torch.Tensor):
        # x: inputs, w: MCMC weights
        mu_x_reconstructed, mu, logvar = self(x)

        # Reconstruction loss
        reconstruction_loss = 0.5 * (
            torch.sum(
                self.log_input_var
                + torch.exp(-self.log_input_var) * (mu_x_reconstructed - x) ** 2,
                dim=-1,
            )
        )
        reconstruction_loss = torch.mean(w * reconstruction_loss)

        # KL divergence
        # KL(N(mu, exp(logvar)) || N(self.latent_mu, exp(self.latent_logvar)))
        latent_var_inv = torch.exp(-self.latent_logvar)
        var = torch.exp(logvar)

        # https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
        kl_divergence = 0.5 * (
            torch.sum(self.latent_logvar - logvar, dim=-1)
            + torch.sum(latent_var_inv * (mu - self.latent_mu) ** 2, dim=-1)
            + torch.sum(latent_var_inv * var, dim=-1)
            - logvar.shape[-1]
        )

        kl_divergence = torch.mean(w * kl_divergence)

        # if torch.isnan(reconstruction_loss) or torch.isnan(kl_divergence):
        #     print(mu_x_reconstructed)
        #     print(mu)
        #     print(logvar)
        #     print(reconstruction_loss)
        #     print(kl_divergence)
        #     print(self.__loss_weights)

        #     raise ValueError("NaN loss")

        return reconstruction_loss, kl_divergence

    def training_step(self, batch: torch.Tensor, batch_idx):
        x, w = batch
        x, w = x.to(self.device), w.to(self.device)
        reconstruction_loss, kl_divergence = self.recons_and_kl(x, w)
        loss = (1 - self.beta) * reconstruction_loss + self.beta * kl_divergence

        self.train_losses = torch.cat(
            (self.train_losses, loss.detach().to("cpu").view(1))
        )
        self.train_batchsizes = torch.cat(
            (self.train_batchsizes, torch.tensor(x.shape[0]).view(1))
        )

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx):
        x, w = batch
        x, w = x.to(self.device), w.to(self.device)
        reconstruction_loss, kl_divergence = self.recons_and_kl(x, w)
        loss = (1 - self.beta) * reconstruction_loss + self.beta * kl_divergence

        self.validation_losses = torch.cat(
            (self.validation_losses, loss.detach().to("cpu").view(1))
        )
        self.validation_kls = torch.cat(
            (self.validation_kls, kl_divergence.detach().to("cpu").view(1))
        )
        self.validation_reconstruction_losses = torch.cat(
            (
                self.validation_reconstruction_losses,
                reconstruction_loss.detach().to("cpu").view(1),
            )
        )
        self.validation_batchsizes = torch.cat(
            (self.validation_batchsizes, torch.tensor(x.shape[0]).view(1))
        )
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.sum(self.train_losses * self.train_batchsizes) / torch.sum(
            self.train_batchsizes
        )
        self.log("train_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_losses = torch.tensor([])
        self.train_batchsizes = torch.tensor([])

    def on_validation_epoch_end(self):
        avg_loss = torch.sum(
            self.validation_losses * self.validation_batchsizes
        ) / torch.sum(self.validation_batchsizes)
        avg_kl = torch.sum(
            self.validation_kls * self.validation_batchsizes
        ) / torch.sum(self.validation_batchsizes)
        avg_reconstruction_loss = torch.sum(
            self.validation_reconstruction_losses * self.validation_batchsizes
        ) / torch.sum(self.validation_batchsizes)
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_kl", avg_kl, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_reconstruction_loss",
            avg_reconstruction_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        self.validation_losses = torch.tensor([])
        self.validation_kls = torch.tensor([])
        self.validation_reconstruction_losses = torch.tensor([])
        self.validation_batchsizes = torch.tensor([])

        # save the model if the validation loss is the lowest so far

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def fit(self, train_dataset, val_dataset, batch_size, max_epochs=100):
        logger = TensorBoardLogger("tb_logs", name=self.run_name)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[self.checkpoint_callback],
            logger=logger,
        )

        self.__loss_weights = self.__loss_weights.to(self.device)
        trainer.fit(
            self,
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        )

        # collect first tensor iterating over val_dataset
        val_x = torch.tensor([])
        for x, _ in val_dataset:
            val_x = torch.cat((val_x, x.view(1, -1)), dim=0)

        logger.experiment.add_figure(
            "latent_space", self.plot_result(val_x), self.current_epoch
        )

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def plot_result(self, val_x: torch.Tensor):
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

        f1 = 0
        f2 = 1

        val_data_encoded = self.encode(val_x)
        val_data_mu, val_data_logvar = val_data_encoded
        val_data_decoded = self.decode(val_data_mu)

        val_data_encoded_np = val_data_mu.detach().cpu().numpy()
        val_data_decoded_np = val_data_decoded.detach().cpu().numpy()
        val_data_np = val_x.detach().cpu().numpy()

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

        ax[0].scatter(val_data_encoded_np[:, 0], val_data_encoded_np[:, 1], s=1)
        ax[0].set_title("Encoded latent space")

        xmin, xmax = val_data_decoded_np[:, f1].min(), val_data_decoded_np[:, f1].max()
        xmin, xmax = min(xmin, val_data_np[:, f1].min()), max(
            xmax, val_data_np[:, f1].max()
        )
        ymin, ymax = val_data_decoded_np[:, f2].min(), val_data_decoded_np[:, f2].max()
        ymin, ymax = min(ymin, val_data_np[:, f2].min()), max(
            ymax, val_data_np[:, f2].max()
        )
        x_diff = ymax - ymin
        y_diff = xmax - xmin
        ymin, ymax = ymin - 0.1 * x_diff, ymax + 0.1 * x_diff
        xmin, xmax = xmin - 0.1 * y_diff, xmax + 0.1 * y_diff

        ax[1].scatter(val_data_decoded_np[:, f1], val_data_decoded_np[:, f2], s=1)
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylim(ymin, ymax)
        ax[1].set_title("Decoded latent space")
        ax[1].set_xlabel(features[f1])
        ax[1].set_ylabel(features[f2])

        ax[2].scatter(val_data_np[:, f1], val_data_np[:, f2], s=1)
        ax[2].set_xlim(xmin, xmax)
        ax[2].set_ylim(ymin, ymax)
        ax[2].set_title("Original space")
        ax[2].set_xlabel(features[f1])
        ax[2].set_ylabel(features[f2])

        plt.tight_layout()
        plt.close()

        return fig

    def log_config(self, config: dict):
        self.logger.experiment.add_text("config", str(config), self.current_epoch)
