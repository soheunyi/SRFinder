import os
from typing import Literal
import torch
import tqdm
from fvt_encoder import FvTEncoder
import pytorch_lightning as pl
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar, Callback
import pathlib
from scipy.optimize import minimize

from network_blocks import conv1d
from training_info import TrainingInfoV2


def kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    h: float,
    kernel_type: Literal["dirichlet", "gaussian", "bins", "nbd", "knn"] = "gaussian",
):
    x = x.reshape(-1, 1)
    y = y.reshape(1, -1)

    if kernel_type == "dirichlet":
        alpha = y / h + 1
        beta = (1 - y) / h + 1
        return (
            (x ** (alpha - 1))
            * ((1 - x) ** (beta - 1))
            * torch.exp(
                torch.lgamma(torch.tensor(1 / h + 2))
                - torch.lgamma(alpha)
                - torch.lgamma(beta)
            )
        )
    elif kernel_type == "gaussian":
        return torch.exp(-((x - y) ** 2) / h**2)
    elif kernel_type == "bins":
        bins = torch.linspace(0, 1, int(1 / h) + 1).to(x.device)
        return (torch.bucketize(x, bins) == torch.bucketize(y, bins)).to(torch.float32)
    elif kernel_type == "nbd":
        return ((x - y).abs() < h).to(torch.float32)
    elif kernel_type == "knn":
        raise NotImplementedError("KNN kernel not implemented")
    else:
        raise ValueError(f"Invalid kernel type: {kernel_type}")


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
        calibration_h: float = 0.05,
        calibration_beta: float = 0.0,
        calibration_p: float = 1,
        calibration_kernel_type: Literal[
            "dirichlet", "gaussian", "bins", "nbd", "knn"
        ] = "gaussian",
        cheating_beta: float = 0.0,
        cheating_h: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dim_j = dim_input_jet_features
        self.dim_d = dim_dijet_features
        self.dim_q = dim_quadjet_features

        self.calibration_h = calibration_h
        self.calibration_beta = calibration_beta
        self.calibration_p = calibration_p
        self.calibration_kernel_type = calibration_kernel_type

        self.cheating_beta = cheating_beta
        self.cheating_h = cheating_h

        self.train_losses = torch.tensor([])
        self.train_batchsizes = torch.tensor([])
        self.train_total_weights = 0.0
        self.val_losses = torch.tensor([])
        self.val_batchsizes = torch.tensor([])
        self.val_total_weights = 0.0
        self.best_val_loss = torch.inf

        self.train_cal_losses = torch.tensor([])
        self.val_cal_losses = torch.tensor([])

        self.train_ce_losses = torch.tensor([])
        self.val_ce_losses = torch.tensor([])

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
        self.to(device)

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

    def ce_loss(self, y_logits: torch.Tensor, y_labels: torch.Tensor, reduction="none"):
        # y_pred: logits, y_labels: labels
        return F.cross_entropy(y_logits, y_labels, reduction=reduction)

    def calibration_loss(
        self,
        y_logits: torch.Tensor,
        y_labels: torch.Tensor,
        weights: torch.Tensor,
        p: float = 1,
    ):
        preds = torch.sigmoid(y_logits[:, 1] - y_logits[:, 0])
        K = kernel(preds, preds, self.calibration_h, self.calibration_kernel_type)
        # K = K.fill_diagonal_(0.0)
        K = K * weights.reshape(1, -1)  # is this correct?
        K = K / K.sum(dim=1, keepdim=True)
        acc = (K @ y_labels.to(torch.float32).reshape(-1, 1)).reshape(-1)

        return (acc - preds).abs() ** p

    def cheating_loss(
        self, y_logits: torch.Tensor, y_labels: torch.Tensor, weights: torch.Tensor
    ):
        preds = torch.sigmoid(y_logits[:, 1] - y_logits[:, 0])
        bins = torch.linspace(0, 1, int(1 / self.calibration_h) + 1).to(y_logits.device)
        bin_idx = torch.bucketize(preds, bins)
        bin_idx = torch.clip(bin_idx, 1, len(bins) - 1) - 1

        reweighted = torch.where(y_labels == 1, -1, preds / (1 - preds)) * weights

        # return reweighted.sum() ** 2 / weights.sum()
        return torch.tensor(0.0)  # disable cheating loss

    def loss(self, y_logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
        ce_loss = self.ce_loss(y_logits, y, reduction="none")
        loss = ce_loss + self.calibration_beta * (
            self.calibration_loss(y_logits, y, w, p=self.calibration_p)
        ) ** (1 / self.calibration_p)
        loss = (loss * w).mean()
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx
    ):
        x, y, w = batch  # x: features, y: labels, w: weights
        x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
        y_logits = self(x)
        ce_loss = (self.ce_loss(y_logits, y, reduction="none") * w).mean()
        cal_loss = (self.calibration_loss(y_logits, y, w) * w).mean()
        loss = ce_loss + self.calibration_beta * cal_loss

        self.train_losses = torch.cat(
            (self.train_losses, loss.detach().to("cpu").view(1))
        )
        self.train_cal_losses = torch.cat(
            (self.train_cal_losses, cal_loss.detach().to("cpu").view(1))
        )
        self.train_ce_losses = torch.cat(
            (self.train_ce_losses, ce_loss.detach().to("cpu").view(1))
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
        cal_loss = (self.calibration_loss(y_logits, y, w) * w).mean()
        ce_loss = (self.ce_loss(y_logits, y, reduction="none") * w).mean()
        loss = ce_loss + self.calibration_beta * cal_loss

        self.val_losses = torch.cat((self.val_losses, loss.detach().to("cpu").view(1)))
        self.val_cal_losses = torch.cat(
            (self.val_cal_losses, cal_loss.detach().to("cpu").view(1))
        )
        self.val_ce_losses = torch.cat(
            (self.val_ce_losses, ce_loss.detach().to("cpu").view(1))
        )
        self.val_batchsizes = torch.cat(
            (self.val_batchsizes, torch.tensor(x.shape[0]).view(1))
        )
        self.val_total_weights += w.sum()

        return loss

    def on_train_epoch_end(self):
        avg_loss = (
            torch.sum(self.train_losses * self.train_batchsizes)
            / self.train_total_weights
        )
        avg_cal_loss = (
            torch.sum(self.train_cal_losses * self.train_batchsizes)
            / self.train_total_weights
        )
        avg_ce_loss = (
            torch.sum(self.train_ce_losses * self.train_batchsizes)
            / self.train_total_weights
        )

        self.log("train_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("train_ce_loss", avg_ce_loss, on_epoch=True, prog_bar=True)
        self.log(
            "train_cal_loss",
            avg_cal_loss,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_losses = torch.tensor([])
        self.train_cal_losses = torch.tensor([])
        self.train_ce_losses = torch.tensor([])
        self.train_cheating_losses = torch.tensor([])
        self.train_batchsizes = torch.tensor([])
        self.train_total_weights = 0.0

        self.nan_check()

    def on_validation_epoch_end(self):
        avg_loss = (
            torch.sum(self.val_losses * self.val_batchsizes) / self.val_total_weights
        )
        avg_cal_loss = (
            torch.sum(self.val_cal_losses * self.val_batchsizes)
            / self.val_total_weights
        )
        avg_ce_loss = (
            torch.sum(self.val_ce_losses * self.val_batchsizes) / self.val_total_weights
        )
        self.log(
            "val_loss",
            avg_loss,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_ce_loss",
            avg_ce_loss,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_cal_loss",
            avg_cal_loss,
            on_epoch=True,
            prog_bar=True,
        )

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        self.val_losses = torch.tensor([])
        self.val_cal_losses = torch.tensor([])
        self.val_ce_losses = torch.tensor([])
        self.val_cheating_losses = torch.tensor([])
        self.val_batchsizes = torch.tensor([])
        self.val_total_weights = 0.0

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
        callbacks: list[Callback] = [],
    ):
        if train_seed is not None:
            pl.seed_everything(train_seed)

        torch.set_float32_matmul_precision("medium")

        logger = TensorBoardLogger("tb_logs", name=self.run_name)

        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
        )
        progress_bar = TQDMProgressBar(
            refresh_rate=max(1, (len(train_dataset) // batch_size) // 10)
        )
        callbacks = callbacks + [early_stop_callback, progress_bar]

        if save_checkpoint:
            # raise error if checkpoint file exists
            checkpoint_path = pathlib.Path(
                f"./data/checkpoints/{self.run_name}_best.ckpt"
            )
            if checkpoint_path.exists():
                raise FileExistsError(f"{checkpoint_path} already exists")

            checkpoint_callback = ModelCheckpoint(
                dirpath="./data/checkpoints/",
                filename=f"{self.run_name}_best",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            )
            callbacks.append(checkpoint_callback)

        tmp_checkpoint_path = pathlib.Path(
            f"./data/tmp/checkpoints/{self.run_name}_best.ckpt"
        )
        # delete tmp checkpoint if it exists
        if tmp_checkpoint_path.exists():
            print(f"Deleting existing tmp checkpoint: {tmp_checkpoint_path}")
            os.remove(tmp_checkpoint_path)

        tmp_checkpoint_callback = ModelCheckpoint(
            dirpath="./data/tmp/checkpoints/",
            filename=f"{self.run_name}_best",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        callbacks.append(tmp_checkpoint_callback)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
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

    def predict(self, x: torch.Tensor, do_tqdm=False):
        with torch.no_grad():
            x_dataloader = DataLoader(x, batch_size=1024, shuffle=False)
            y_pred = torch.tensor([])
            if do_tqdm:
                x_dataloader = tqdm.tqdm(x_dataloader)
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

    def representations(
        self, x: torch.Tensor, do_tqdm=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x_dataloader = DataLoader(x, batch_size=1024, shuffle=False)

            x_dataloader = x_dataloader if not do_tqdm else tqdm.tqdm(x_dataloader)
            q_repr = torch.tensor([])
            view_scores = torch.tensor([])

            for batch in x_dataloader:
                batch = batch.to(self.device)
                q = self.encoder(batch)
                q_repr = torch.cat((q_repr, q.to("cpu")), dim=0)

                view_scores_batch = self.select_q(q)
                view_scores_batch = F.softmax(view_scores_batch, dim=-1)
                view_scores = torch.cat(
                    (view_scores, view_scores_batch.to("cpu")), dim=0
                )

        return q_repr, view_scores

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

    def freeze_encoder(self, freeze=True):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def load_tmp_checkpoint(self):
        checkpoint_path = pathlib.Path(
            f"./data/tmp/checkpoints/{self.run_name}_best.ckpt"
        )
        self = FvTClassifier.load_from_checkpoint(checkpoint_path)
        self.eval()
        self.to(self.device)

    # def on_fit_end(self):
    #     print("Loading tmp checkpoint")
    #     self.load_tmp_checkpoint()
    #     self.eval()
    #     self.to(self.device)
