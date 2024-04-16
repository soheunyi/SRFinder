import torch
from torch import optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Literal


class Activation(nn.Module):
    def __init__(
        self, activation: Literal["ReLU", "RReLU", "LeakyReLU", "SiLU", "ELU"]
    ):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        if self.activation == "ReLU":
            return F.relu(x)
        elif self.activation == "RReLU":
            return F.rrelu(x)
        elif self.activation == "LeakyReLU":
            return F.leaky_relu(x)
        elif self.activation == "SiLU":
            return F.silu(x)
        elif self.activation == "ELU":
            return F.elu(x)
        else:
            raise NotImplementedError


class ResNetBlock(nn.Module):
    # ResNet block with MLP
    def __init__(
        self,
        in_channels,
        out_channels,
        bias: bool = True,
        activation: (
            Literal["ReLU", "RReLU", "LeakyReLU", "SiLU", "ELU"] | None
        ) = "SiLU",
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            Activation(activation) if activation is not None else nn.Identity(),
        )

    def forward(self, x):
        return x + self.mlp(x)


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        activation: Literal["ReLU", "RReLU", "LeakyReLU", "SiLU", "ELU"] = "SiLU",
        last_bias: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        layer_dims = [input_dim] + hidden_dims + [latent_dim]
        # linear layers
        self.encoder = nn.Sequential(
            *(
                [
                    nn.Sequential(
                        nn.Linear(layer_dims[i], layer_dims[i + 1]),
                        Activation(activation),
                    )
                    for i in range(len(layer_dims) - 2)
                ]
                + [
                    nn.Linear(layer_dims[-2], layer_dims[-1], bias=last_bias),
                ]
            )
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: Literal["ReLU", "RReLU", "LeakyReLU", "SiLU", "ELU"] = "SiLU",
        last_bias: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        layer_dims = [latent_dim] + hidden_dims + [output_dim]
        # linear layers
        self.decoder = nn.Sequential(
            *(
                [
                    nn.Sequential(
                        nn.Linear(layer_dims[i], layer_dims[i + 1]),
                        Activation(activation),
                    )
                    for i in range(len(layer_dims) - 2)
                ]
                + [
                    nn.Linear(layer_dims[-2], layer_dims[-1], bias=last_bias),
                ]
            )
        )

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        run_name_base: str,
        activation: Literal["ReLU", "RReLU", "LeakyReLU", "SiLU", "ELU"] = "SiLU",
        lr: float = 1e-3,
        loss_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.encoder = Encoder(input_dim, latent_dim, hidden_dims, activation)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dims[::-1], activation)

        self.__loss_weights = None

        if loss_weights is not None:
            assert loss_weights.shape[0] == input_dim
            self.__loss_weights = loss_weights

        self.lr = lr
        self.train_losses = torch.tensor([])
        self.validation_losses = torch.tensor([])
        self.train_batchsizes = torch.tensor([])
        self.validation_batchsizes = torch.tensor([])

        self.best_val_loss = torch.inf
        self.checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/",
            filename=f"{run_name_base}_best",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )

    @property
    def loss_weights(self):
        return self.__loss_weights

    def set_loss_weights(self, loss_weights: torch.Tensor):
        if self.__loss_weights is not None:
            raise ValueError("Loss weights are already set")
        assert loss_weights.shape[0] == self.input_dim
        self.__loss_weights = loss_weights.detach()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch: torch.Tensor, batch_idx):
        x = batch.to(self.device)
        x_hat = self.forward(x)
        loss = torch.sum(self.loss_weights * (x_hat - x) ** 2) / torch.sum(
            self.loss_weights
        )

        self.train_losses = torch.cat(
            (self.train_losses, loss.detach().to("cpu").view(1))
        )
        self.train_batchsizes = torch.cat(
            (self.train_batchsizes, torch.tensor(x.shape[0]).view(1))
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx):
        x = batch.to(self.device)
        x_hat = self.forward(x)
        loss = torch.sum(self.loss_weights * (x_hat - x) ** 2) / torch.sum(
            self.loss_weights
        )
        self.validation_losses = torch.cat(
            (self.validation_losses, loss.detach().to("cpu").view(1))
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
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        self.validation_losses = torch.tensor([])
        self.validation_batchsizes = torch.tensor([])

        # save the model if the validation loss is the lowest so far

    def on_fit_end(self):
        # save a figure about training and validation loss
        pass

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def fit(self, train_dataset, val_dataset, batch_size, max_epochs=100):
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[self.checkpoint_callback],
        )

        self.__loss_weights = self.__loss_weights.to(self.device)
        trainer.fit(
            self,
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
