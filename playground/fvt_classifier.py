import torch
from fvt_encoder import FvTEncoder
import pytorch_lightning as pl
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import pathlib


from network_blocks import conv1d


class FvTClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        dim_input_jet_features,
        dim_dijet_features,
        dim_quadjet_features,
        run_name: str,
        device: str = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dim_j = dim_input_jet_features
        self.dim_d = dim_dijet_features
        self.dim_q = dim_quadjet_features

        self.train_losses = torch.tensor([])
        self.train_batchsizes = torch.tensor([])
        self.train_total_weights = 0.0
        self.validation_losses = torch.tensor([])
        self.validation_batchsizes = torch.tensor([])
        self.validation_total_weights = 0.0
        self.best_val_loss = torch.inf

        self.num_classes = num_classes
        self.lr = lr
        self.run_name = run_name

        self.encoder = FvTEncoder(
            dim_input_jet_features=dim_input_jet_features,
            dim_dijet_features=dim_dijet_features,
            dim_quadjet_features=dim_quadjet_features,
            device=device,
        )

        self.select_q = conv1d(
            dim_quadjet_features, 1, 1, name="quadjet selector", batchNorm=True
        )

        self.out = conv1d(
            dim_quadjet_features, num_classes, 1, name="out", batchNorm=True
        )

    def forward(self, x: torch.Tensor):
        n = x.shape[0]
        q = self.encoder(x)
        q_score = self.select_q(q)
        q_score = F.softmax(q_score, dim=-1)
        event = torch.matmul(q, q_score.transpose(1, 2))
        event = event.view(n, self.dim_q, 1)

        # project the final event-level pixel into the class score space
        class_score = self.out(event)
        class_score = class_score.view(n, self.num_classes)

        if torch.isnan(class_score).any():
            print("NaN found in forward")
            print("x", x)
            print("q", q)
            print("q_score", q_score)
            print("event", event)
            print("class_score", class_score)

            raise ValueError("NaN found in forward")

        return class_score

    def loss(self, y_logits: torch.Tensor, y_true: torch.Tensor, reduction="none"):
        # y_pred: logits, y_true: labels
        return F.cross_entropy(y_logits, y_true, reduction=reduction)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx
    ):
        x, y, w = batch  # x: features, y: labels, w: weights
        x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
        y_logits = self(x)
        loss = self.loss(y_logits, y, reduction="none")
        loss = (loss * w).mean()

        self.train_losses = torch.cat(
            (self.train_losses, loss.detach().to("cpu").view(1))
        )
        self.train_batchsizes = torch.cat(
            (self.train_batchsizes, torch.tensor(x.shape[0]).view(1))
        )
        self.train_total_weights += w.sum()

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx
    ):
        x, y, w = batch
        x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
        y_logits = self(x)
        loss = self.loss(y_logits, y, reduction="none")
        loss = (loss * w).mean()

        self.validation_losses = torch.cat(
            (self.validation_losses, loss.detach().to("cpu").view(1))
        )
        self.validation_batchsizes = torch.cat(
            (self.validation_batchsizes, torch.tensor(x.shape[0]).view(1))
        )
        self.validation_total_weights += w.sum()

        return loss

    def on_train_epoch_end(self):
        avg_loss = (
            torch.sum(self.train_losses * self.train_batchsizes)
            / self.train_total_weights
        )

        self.log("train_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.train_losses = torch.tensor([])
        self.train_batchsizes = torch.tensor([])
        self.train_total_weights = 0.0

        self.nan_check()

    def on_validation_epoch_end(self):
        avg_loss = (
            torch.sum(self.validation_losses * self.validation_batchsizes)
            / self.validation_total_weights
        )
        self.log(
            "val_loss",
            avg_loss,
            on_epoch=True,
            prog_bar=True,
        )
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        self.validation_losses = torch.tensor([])
        self.validation_batchsizes = torch.tensor([])
        self.validation_total_weights = 0.0

        self.nan_check()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def fit(
        self,
        train_dataset,
        val_dataset,
        batch_size,
        max_epochs=30,
        train_seed: int = None,
        save_checkpoint: bool = True,
    ):
        if train_seed is not None:
            pl.seed_everything(train_seed)

        logger = TensorBoardLogger("tb_logs", name=self.run_name)
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
        )

        callbacks = [early_stop_callback]

        if save_checkpoint:
            # raise error if checkpoint file exists
            checkpoint_path = pathlib.Path(f"checkpoints/{self.run_name}_best.ckpt")
            if checkpoint_path.exists():
                raise FileExistsError(f"{checkpoint_path} already exists")

            checkpoint_callback = ModelCheckpoint(
                dirpath="checkpoints/",
                filename=f"{self.run_name}_best",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            )
            callbacks.append(checkpoint_callback)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
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

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            x_dataloader = DataLoader(x, batch_size=1024)
            y_pred = torch.tensor([])
            for batch in x_dataloader:
                batch = batch.to(self.device)
                logit = self(batch)
                pred = F.softmax(logit, dim=-1)
                if torch.isnan(pred).any():
                    print("NaN found in prediction")
                    print(pred)
                    print(logit)
                    raise ValueError("NaN found in prediction")

                y_pred = torch.cat((y_pred, pred.to("cpu")), dim=0)

        return y_pred

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def nan_check(self):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(name, "has NaN")
                raise ValueError("NaN or inf found in model parameters")
            if torch.isinf(param).any():
                print(name, "has inf")
                raise ValueError("NaN or inf found in model parameters")
        return
