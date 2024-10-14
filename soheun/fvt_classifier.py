import os
from typing import Literal
import numpy as np
import torch
import tqdm
from fvt_encoder import FvTEncoder
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar,
    Callback,
)
from pytorch_lightning.loggers import TensorBoardLogger
import pathlib

from utils import require_keys
from attention_classifier import AttentionClassifier
from data_modules import FvTDataModule


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
        depth: dict = {
            "encoder": 4,
            "decoder": 1,
        },
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dim_j = dim_input_jet_features
        self.dim_d = dim_dijet_features
        self.dim_q = dim_quadjet_features

        self.num_classes = num_classes
        self.run_name = run_name
        require_keys(depth, ["encoder", "decoder"])
        self.depth = depth

        self.optimizer_config = None
        self.lr_scheduler_config = None

        self.train_losses = torch.tensor([])
        self.train_batchsizes = torch.tensor([])
        self.train_total_weights = 0.0
        self.val_losses = torch.tensor([])
        self.val_batchsizes = torch.tensor([])
        self.val_total_weights = 0.0
        self.best_val_loss = torch.inf

        self.val_preds = torch.tensor([])
        self.val_labels = torch.tensor([])
        self.val_weights = torch.tensor([])

        self.history: list[dict] = []

        self.encoder = FvTEncoder(
            dim_input_jet_features=dim_input_jet_features,
            dim_dijet_features=dim_dijet_features,
            dim_quadjet_features=dim_quadjet_features,
            device=device,
            depth=depth["encoder"],
        )

        self.attention_classifier = AttentionClassifier(
            dim_quadjet_features, num_classes, depth["decoder"]
        )
        self.to(device)

    def forward(self, x: torch.Tensor):
        q = self.encoder(x)
        class_score = self.attention_classifier(q)

        if torch.isnan(class_score).any():
            print("NaN found in forward")
            print("x", x)
            print("q", q)
            print("class_score", class_score)

            raise ValueError("NaN found in forward")

        return class_score

    def ce_loss(self, y_logits: torch.Tensor, y_labels: torch.Tensor, reduction="none"):
        # y_pred: logits, y_labels: labels
        return F.cross_entropy(y_logits, y_labels, reduction=reduction)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx
    ):
        x, y, w = batch  # x: features, y: labels, w: weights
        x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
        y_logits = self(x)
        loss = (self.ce_loss(y_logits, y, reduction="none") * w).mean()

        self.train_losses = torch.cat(
            (self.train_losses, loss.detach().to("cpu").view(1))
        )
        self.train_batchsizes = torch.cat(
            (self.train_batchsizes, torch.tensor(x.shape[0]).view(1))
        )
        self.train_total_weights += w.sum().item()

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx
    ):
        x, y, w = batch
        x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
        y_logits = self(x)
        loss = (self.ce_loss(y_logits, y, reduction="none") * w).mean()
        self.val_losses = torch.cat((self.val_losses, loss.detach().to("cpu").view(1)))
        self.val_batchsizes = torch.cat(
            (self.val_batchsizes, torch.tensor(x.shape[0]).view(1))
        )
        self.val_total_weights += w.sum().item()

        preds = torch.sigmoid(y_logits[:, 1] - y_logits[:, 0])

        self.val_preds = torch.cat((self.val_preds, preds.to("cpu")))
        self.val_labels = torch.cat((self.val_labels, y.to("cpu")))
        self.val_weights = torch.cat((self.val_weights, w.to("cpu")))

        return loss

    def on_train_epoch_start(self):
        self.log("step", self.trainer.current_epoch, on_epoch=True)

    def on_validation_epoch_start(self):
        self.log("step", self.trainer.current_epoch, on_epoch=True)

    def on_train_epoch_end(self):
        avg_loss = (
            torch.sum(self.train_losses * self.train_batchsizes)
            / self.train_total_weights
        )
        avg_loss_first_digits = (
            int(avg_loss.item() * 1000) if not torch.isnan(avg_loss) else 0
        )
        avg_loss_second_digits = (
            avg_loss.item() * 1000 - int(avg_loss.item() * 1000)
            if not torch.isnan(avg_loss)
            else 0
        )
        self.log("train_loss", avg_loss, on_epoch=True)
        self.log(
            "train_loss_lower_digits",
            avg_loss_first_digits,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_loss_second_digits",
            avg_loss_second_digits,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_losses = torch.tensor([])
        self.train_batchsizes = torch.tensor([])
        self.train_total_weights = 0.0

        self.nan_check()

    def on_validation_epoch_end(self):
        avg_loss = (
            torch.sum(self.val_losses * self.val_batchsizes) / self.val_total_weights
        )
        reweights = self.val_preds / (1 - self.val_preds)
        reweights = torch.where(
            self.val_labels == 1, 1, -self.val_preds / (1 - self.val_preds)
        )

        nbins = 10
        q = torch.linspace(0, 1, nbins + 1).to(self.val_preds.device)
        bins = torch.quantile(self.val_preds, q)
        bin_idx = torch.bucketize(self.val_preds, bins)
        bin_idx = torch.clip(bin_idx, 1, nbins) - 1

        w_rw = reweights * self.val_weights
        w_rw_sq = reweights**2 * self.val_weights

        bin_losses = []
        for bin_i in range(nbins):
            bin_mask = bin_idx == bin_i
            bin_losses.append(
                torch.sum(w_rw[bin_mask]) ** 2 / torch.sum(w_rw_sq[bin_mask])
            )

        avg_sigma_sq = torch.mean(torch.tensor(bin_losses))
        avg_loss_first_digits = (
            int(avg_loss.item() * 1000) if not torch.isnan(avg_loss) else 0
        )
        avg_loss_second_digits = (
            avg_loss.item() * 1000 - int(avg_loss.item() * 1000)
            if not torch.isnan(avg_loss)
            else 0
        )
        last_lr = (
            self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[-1]
            if self.lr_scheduler_config["type"] != "none"
            else self.optimizer_config["lr"]
        )

        self.log(
            "val_loss",
            avg_loss.item(),
            on_epoch=True,
        )
        self.log(
            "1000x_val_loss_first_digits",
            avg_loss_first_digits,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "1000x_val_loss_second_digits",
            avg_loss_second_digits,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_sigma_sq",
            avg_sigma_sq.item(),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "lr",
            last_lr,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "batch_size",
            self.datamodule.batch_size,
            on_epoch=True,
        )

        tb_logs = {
            "val_loss": avg_loss,
            "val_sigma_sq": avg_sigma_sq,
            "lr": last_lr,
            "batch_size": self.datamodule.batch_size,
        }

        self.update_history(tb_logs)

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        self.val_losses = torch.tensor([])
        self.val_sigma_sqes = torch.tensor([])
        self.val_batchsizes = torch.tensor([])
        self.val_total_weights = 0.0
        self.val_preds = torch.tensor([])
        self.val_labels = torch.tensor([])
        self.val_weights = torch.tensor([])

        self.nan_check()

    def update_history(self, kv: dict[str, float]):
        saved_epochs = [h["epoch"] for h in self.history]
        if len(np.unique(saved_epochs)) != len(saved_epochs):
            raise ValueError("Duplicate epoch found in history")

        if self.current_epoch not in saved_epochs:
            self.history.append({"epoch": self.current_epoch, **kv})
        else:
            for h in self.history:
                if self.current_epoch == h.get("epoch", -1):
                    h.update(kv)
                    break

    def configure_optimizers(self):
        assert self.optimizer_config is not None
        assert self.lr_scheduler_config is not None

        require_keys(self.optimizer_config, ["type", "lr"])
        require_keys(self.lr_scheduler_config, ["type"])

        if self.lr_scheduler_config["type"] == "ReduceLROnPlateau":
            require_keys(
                self.lr_scheduler_config,
                ["factor", "threshold", "patience", "cooldown", "min_lr"],
            )

        if self.optimizer_config["type"] == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=self.optimizer_config["lr"])
        elif self.optimizer_config["type"] == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.optimizer_config["lr"])
        else:
            raise ValueError(f"Invalid optimizer type: {self.optimizer_config['type']}")

        return_dict = {"optimizer": optimizer, "monitor": "val_loss"}

        if self.lr_scheduler_config["type"] == "none":
            pass
        elif self.lr_scheduler_config["type"] == "ReduceLROnPlateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.lr_scheduler_config["factor"],
                threshold=self.lr_scheduler_config["threshold"],
                patience=self.lr_scheduler_config["patience"],
                cooldown=self.lr_scheduler_config["cooldown"],
                min_lr=self.lr_scheduler_config["min_lr"],
            )
            return_dict["lr_scheduler"] = lr_scheduler
        else:
            raise ValueError(
                f"Invalid lr scheduler type: {self.lr_scheduler_config['type']}"
            )

        return return_dict

    def fit(
        self,
        train_dataset,
        val_dataset,
        max_epochs=50,
        train_seed: int = None,
        save_checkpoint: bool = True,
        callbacks: list[Callback] = [],
        tb_log_dir: str = "tmp",
        optimizer_config: dict = {},
        lr_scheduler_config: dict = {},
        early_stop_patience: None | int = None,
        dataloader_config: dict = {},
    ):
        assert "batch_size" in dataloader_config

        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.early_stop_patience = early_stop_patience

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.dataloader_config = dataloader_config

        if train_seed is not None:
            pl.seed_everything(train_seed)

        torch.set_float32_matmul_precision("medium")

        tb_log_dir = pathlib.Path(f"./tb_logs/{tb_log_dir}")
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        logger = TensorBoardLogger(tb_log_dir, name=self.run_name)

        progress_bar = TQDMProgressBar(
            refresh_rate=max(
                1, (len(train_dataset) // dataloader_config["batch_size"]) // 10
            )
        )
        callbacks = callbacks + [progress_bar]

        if self.early_stop_patience is not None:
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=self.early_stop_patience,
                verbose=False,
                mode="min",
            )
            callbacks.append(early_stop_callback)

        if save_checkpoint:
            delete_existing_checkpoints = False
            checkpoint_dir = pathlib.Path(f"./data/checkpoints/")
        else:
            delete_existing_checkpoints = True
            checkpoint_dir = pathlib.Path(f"./data/tmp/checkpoints/")

        for mode in ["best", "last"]:
            ckpt_path = checkpoint_dir / f"{self.run_name}_{mode}.ckpt"
            if ckpt_path.exists():
                if delete_existing_checkpoints:
                    print(f"Deleting existing checkpoint: {ckpt_path}")
                    os.remove(ckpt_path)
                else:
                    raise FileExistsError(f"{ckpt_path} already exists")

            filename = f"{self.run_name}_{mode}"
            if mode == "best":
                ckpt_callback = ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename=filename,
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                )
                callbacks.append(ckpt_callback)
            elif mode == "last":
                ckpt_callback = ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename=filename,
                    save_last=True,
                )
                callbacks.append(ckpt_callback)
            else:
                raise ValueError(f"Invalid checkpoint mode: {mode}")

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            logger=logger,
            reload_dataloaders_every_n_epochs=1,
        )

        torch.autograd.set_detect_anomaly(True)

        self.datamodule = FvTDataModule(
            train_dataset,
            val_dataset,
            dataloader_config["batch_size"],
            num_workers=4,
            batch_size_milestones=dataloader_config.get("batch_size_milestones", []),
            batch_size_multiplier=dataloader_config.get("batch_size_multiplier", 2),
        )
        trainer.fit(self, datamodule=self.datamodule)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, do_tqdm=False):
        self.eval()
        batch_size = min(2**14, x.shape[0])
        x_dataloader = DataLoader(x, batch_size=batch_size, shuffle=False)
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

    @torch.no_grad()
    def representations(
        self, x: torch.Tensor, do_tqdm=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_dataloader = DataLoader(x, batch_size=min(2**14, x.shape[0]), shuffle=False)

        x_dataloader = x_dataloader if not do_tqdm else tqdm.tqdm(x_dataloader)
        q_repr = torch.tensor([])
        view_scores = torch.tensor([])

        for batch in x_dataloader:
            batch = batch.to(self.device)
            q = self.encoder(batch)
            q_repr = torch.cat((q_repr, q.to("cpu")), dim=0)

            view_scores_batch = self.attention_classifier.select_q(q)
            view_scores_batch = F.softmax(view_scores_batch, dim=-1)
            view_scores = torch.cat((view_scores, view_scores_batch.to("cpu")), dim=0)

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
        print("Loading tmp checkpoint")
        checkpoint_path = pathlib.Path(
            f"./data/tmp/checkpoints/{self.run_name}_best.ckpt"
        )
        self = FvTClassifier.load_from_checkpoint(checkpoint_path)
        self.eval()
        self.to(self.device)
