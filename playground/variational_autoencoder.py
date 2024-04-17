# Variational autoencoder

from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from autoencoder import Activation
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt

from parsers import parse_decoder, parse_encoder, parse_likelihood_and_postprocess
from constants import VIRT_INF, VIRT_ZERO, LOG_VIRT_INF, LOG_VIRT_ZERO


class VariationalAutoencoder(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_config: dict,
        decoder_config: dict,
        likelihood_configs: list[dict],
        run_name: str,
        encoder_activation: Literal[
            "ReLU", "RReLU", "LeakyReLU", "SiLU", "ELU"
        ] = "SiLU",
        lr: float = 1e-3,
        beta: float = 1.0,
    ):
        super().__init__()
        assert 0 <= beta <= 1
        self.save_hyperparameters()

        self.encoder = parse_encoder(encoder_config)
        self.decoder = parse_decoder(decoder_config)

        assert self.encoder.input_dim == input_dim, "Encoder input dim does not match"
        assert (
            self.decoder.input_dim == latent_dim
        ), "Decoder input dim does not match latent dim"

        assert (
            len(likelihood_configs) == input_dim
        ), "Number of likelihoods does not match input dim"
        self.likelihoods = [
            parse_likelihood_and_postprocess(config)[0] for config in likelihood_configs
        ]  # likelihoods
        self.ll_postprocess = [
            parse_likelihood_and_postprocess(config)[1] for config in likelihood_configs
        ]  # activations

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.run_name = run_name
        self.encoder_activation = encoder_activation
        self.lr = lr
        self.beta = beta

        self.train_losses = torch.tensor([])
        self.validation_losses = torch.tensor([])
        self.validation_kls = torch.tensor([])
        self.validation_lls = torch.tensor([])
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

        self.encoder_activation = Activation(encoder_activation)
        self.mu_encoder = nn.Linear(self.encoder.output_dim, latent_dim, bias=False)
        self.logvar_encoder = nn.Linear(self.encoder.output_dim, latent_dim, bias=False)

        # define scale parameter
        self.log_input_var = nn.Parameter(torch.zeros(input_dim))
        self.latent_mu = nn.Parameter(torch.zeros(latent_dim))
        self.latent_logvar = nn.Parameter(torch.zeros(latent_dim))

        # decoders to output parameters of likelihoods
        self.ll_params_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.decoder.output_dim, likelihood.n_params), postprocess
                )
                for likelihood, postprocess in zip(
                    self.likelihoods, self.ll_postprocess
                )
            ]
        )

    def encode(self, x):
        x = self.encoder(x)

        if torch.isnan(x).any():
            print(x)
            raise ValueError("NaN encoder output")

        x = self.encoder_activation(x)

        if torch.isnan(x).any():
            print(x)
            raise ValueError("NaN encoder output")

        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        logvar = torch.clip(logvar, LOG_VIRT_ZERO, LOG_VIRT_INF)

        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        ll_params = [decoder(z) for decoder in self.ll_params_decoders]

        return ll_params

    def reconstruct(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        ll_params = self.decode(z)
        # sample from likelihoods using ll_params
        x_reconstructed = torch.zeros_like(x)
        for i, (likelihood, params) in enumerate(zip(self.likelihoods, ll_params)):
            x_reconstructed[..., i] = likelihood.sample(torch.Size(), params)

        return x_reconstructed

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        if torch.isnan(z).any():
            print("x", x)
            print("mu", mu)
            print("logvar", logvar)
            print("z", z)
            raise ValueError("NaN latent space")

        ll_params = self.decode(z)

        for p in ll_params:
            if torch.isnan(p).any():
                # print all parameters in the model
                for name, param in self.named_parameters():
                    print(name, param)
                print(p)
                raise ValueError("NaN parameters")

        return ll_params, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        # clip std to avoid inf
        noise = torch.randn_like(std)
        return mu + noise * std

    def ll_and_kl(self, x: torch.Tensor, w: torch.Tensor):
        # x: inputs, w: MCMC weights
        ll_params, mu, logvar = self(x)
        ll_values = torch.sum(
            torch.stack(
                [
                    likelihood.log_likelihood(x[..., i], params)
                    for i, (likelihood, params) in enumerate(
                        zip(self.likelihoods, ll_params)
                    )
                ],
                dim=-1,
            ),
            dim=-1,
        )

        ll_avg = torch.mean(w * ll_values)

        # KL divergence
        # KL(N(mu, exp(logvar)) || N(self.latent_mu, exp(self.latent_logvar)))
        latent_logvar = torch.clip(self.latent_logvar, LOG_VIRT_ZERO, LOG_VIRT_INF)
        latent_var_inv = torch.exp(-latent_logvar)

        logvar = torch.clip(logvar, LOG_VIRT_ZERO, LOG_VIRT_INF)  # avoid inf
        var = torch.exp(logvar)

        # https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
        kl_divergence = 0.5 * (
            torch.sum(self.latent_logvar - logvar, dim=-1)
            + torch.sum(latent_var_inv * (mu - self.latent_mu) ** 2, dim=-1)
            + torch.sum(latent_var_inv * var, dim=-1)
            - logvar.shape[-1]
        )

        kl_divergence = torch.mean(w * kl_divergence)

        return ll_avg, kl_divergence

    def training_step(self, batch: torch.Tensor, batch_idx):
        x, w = batch
        x, w = x.to(self.device), w.to(self.device)
        ll_avg, kl_divergence = self.ll_and_kl(x, w)
        loss = (1 - self.beta) * (-ll_avg) + self.beta * kl_divergence

        self.train_losses = torch.cat(
            (self.train_losses, loss.detach().to("cpu").view(1))
        )
        self.train_batchsizes = torch.cat(
            (self.train_batchsizes, torch.tensor(x.shape[0]).view(1))
        )

        # check if gradients are NaN
        if self.check_grad_nan():
            # print all relevant information
            print("x", x)
            print("w", w)
            print("ll_avg", ll_avg)
            print("kl_divergence", kl_divergence)
            print("loss", loss)

            raise ValueError("NaN gradients")

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx):
        x, w = batch
        x, w = x.to(self.device), w.to(self.device)
        ll_avg, kl_divergence = self.ll_and_kl(x, w)
        loss = (1 - self.beta) * (-ll_avg) + self.beta * kl_divergence

        self.validation_losses = torch.cat(
            (self.validation_losses, loss.detach().to("cpu").view(1))
        )
        self.validation_kls = torch.cat(
            (self.validation_kls, kl_divergence.detach().to("cpu").view(1))
        )
        self.validation_lls = torch.cat(
            (
                self.validation_lls,
                ll_avg.detach().to("cpu").view(1),
            )
        )
        self.validation_batchsizes = torch.cat(
            (self.validation_batchsizes, torch.tensor(x.shape[0]).view(1))
        )

        if self.check_grad_nan():
            # print all relevant information
            print("x", x)
            print("w", w)
            print("ll_avg", ll_avg)
            print("kl_divergence", kl_divergence)
            print("loss", loss)

            raise ValueError("NaN gradients")

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
        avg_ll = torch.sum(
            self.validation_lls * self.validation_batchsizes
        ) / torch.sum(self.validation_batchsizes)
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_kl", avg_kl, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_ll",
            avg_ll,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        self.validation_losses = torch.tensor([])
        self.validation_kls = torch.tensor([])
        self.validation_lls = torch.tensor([])
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

        torch.autograd.set_detect_anomaly(True)

        trainer.fit(
            self,
            DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
            ),
            DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            ),
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
        self.training = False

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

        f1 = 10
        f2 = 11

        # memory issue: make DataLoader again

        self.eval()
        val_dataloader = DataLoader(val_x, batch_size=1024, shuffle=False)

        val_data_encoded_np = np.array([])
        val_data_recontstructed_np = np.array([])
        val_data_decoded_mu_np = np.array([])
        val_data_decoded_logvar_np = np.array([])

        for x in val_dataloader:
            encoded = self.encode(x)
            x_encoded_mu = encoded[0].detach().cpu().numpy()
            val_data_encoded_np = (
                x_encoded_mu
                if len(val_data_encoded_np) == 0
                else np.concatenate((val_data_encoded_np, x_encoded_mu), axis=0)
            )

            z = self.reparameterize(*encoded)
            decoded = self.decode(z)
            x_decoded_mu = torch.cat([param[:, [0]] for param in decoded], dim=1)
            X_decoded_logvar = torch.cat([param[:, [1]] for param in decoded], dim=1)
            x_decoded_mu = x_decoded_mu.detach().cpu().numpy()
            X_decoded_logvar = X_decoded_logvar.detach().cpu().numpy()

            val_data_decoded_mu_np = (
                x_decoded_mu
                if len(val_data_decoded_mu_np) == 0
                else np.concatenate((val_data_decoded_mu_np, x_decoded_mu), axis=0)
            )

            val_data_decoded_logvar_np = (
                X_decoded_logvar
                if len(val_data_decoded_logvar_np) == 0
                else np.concatenate(
                    (val_data_decoded_logvar_np, X_decoded_logvar), axis=0
                )
            )

            x_recontstructed = self.reconstruct(x).detach().cpu().numpy()
            val_data_recontstructed_np = (
                x_recontstructed
                if len(val_data_recontstructed_np) == 0
                else np.concatenate(
                    (val_data_recontstructed_np, x_recontstructed), axis=0
                )
            )

        val_data_np = val_x.detach().cpu().numpy()

        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 5))

        ax[0].scatter(val_data_encoded_np[:, 0], val_data_encoded_np[:, 1], s=1)
        ax[0].set_title("Encoded latent space")

        xmin, xmax = (
            val_data_recontstructed_np[:, f1].min(),
            val_data_recontstructed_np[:, f1].max(),
        )
        xmin, xmax = min(xmin, val_data_np[:, f1].min()), max(
            xmax, val_data_np[:, f1].max()
        )
        ymin, ymax = (
            val_data_recontstructed_np[:, f2].min(),
            val_data_recontstructed_np[:, f2].max(),
        )
        ymin, ymax = min(ymin, val_data_np[:, f2].min()), max(
            ymax, val_data_np[:, f2].max()
        )
        x_diff = ymax - ymin
        y_diff = xmax - xmin
        ymin, ymax = ymin - 0.1 * x_diff, ymax + 0.1 * x_diff
        xmin, xmax = xmin - 0.1 * y_diff, xmax + 0.1 * y_diff

        ax[1].scatter(
            val_data_recontstructed_np[:, f1], val_data_recontstructed_np[:, f2], s=1
        )
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylim(ymin, ymax)
        ax[1].set_title("Decoded latent space")
        ax[1].set_xlabel(features[f1])
        ax[1].set_ylabel(features[f2])

        ax[2].scatter(val_data_decoded_mu_np[:, f1], val_data_decoded_mu_np[:, f2], s=1)
        ax[2].set_xlim(xmin, xmax)
        ax[2].set_ylim(ymin, ymax)
        ax[2].set_title("Decoded mu space")
        ax[2].set_xlabel(features[f1])
        ax[2].set_ylabel(features[f2])

        ax[3].scatter(
            np.exp(val_data_decoded_logvar_np / 2)[:, f1],
            np.exp(val_data_decoded_logvar_np / 2)[:, f2],
            s=1,
        )
        ax[3].set_title("Decoded std space")
        ax[3].set_xlabel(features[f1])
        ax[3].set_ylabel(features[f2])

        ax[4].scatter(val_data_np[:, f1], val_data_np[:, f2], s=1)
        ax[4].set_xlim(xmin, xmax)
        ax[4].set_ylim(ymin, ymax)
        ax[4].set_title("Original space")
        ax[4].set_xlabel(features[f1])
        ax[4].set_ylabel(features[f2])

        plt.tight_layout()
        plt.close()

        return fig

    def log_config(self, config: dict):
        self.logger.experiment.add_text("config", str(config), self.current_epoch)

    def check_grad_nan(self):
        for name, param in self.named_parameters():
            if param.grad != None and torch.isnan(param.grad).any():
                print(name, "grad is NaN")
                return True
        return False
