import pathlib
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import tqdm

from network_blocks import ResNetBlock
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar,
    Callback,
)
from pytorch_lightning.loggers import TensorBoardLogger
import torch.optim as optim

from data_modules import FvTDataModule
from utils import require_keys


class AttentionClassifier(pl.LightningModule):
    def __init__(self, dim_q, num_classes, run_name, depth=1):
        super().__init__()
        self.dim_q = dim_q
        self.num_classes = num_classes
        self.depth = depth
        self.run_name = run_name

        self.select_q = ResNetBlock(self.dim_q, 1, self.depth)
        self.out = ResNetBlock(self.dim_q, self.num_classes, self.depth)

        self.train_losses = torch.tensor([])
        self.train_batchsizes = torch.tensor([])
        self.train_total_weights = 0.0
        self.val_losses = torch.tensor([])
        self.val_batchsizes = torch.tensor([])
        self.val_total_weights = 0.0
        self.val_preds = torch.tensor([])
        self.val_labels = torch.tensor([])
        self.val_weights = torch.tensor([])
        self.best_val_loss = torch.inf

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        n = q.shape[0]
        q_score = self.select_q(q)
        q_score = F.softmax(q_score, dim=-1)
        event = torch.matmul(q, q_score.transpose(1, 2))
        event = event.view(n, self.dim_q, 1)
        event = self.out(event)
        event = event.view(n, self.num_classes)
        return event

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        x, y, w = batch
        x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
        y_logits = self(x)
        loss = F.cross_entropy(y_logits, y, reduction="none") * w
        loss = loss.mean()

        self.train_losses = torch.cat([self.train_losses, loss.to("cpu").unsqueeze(0)])
        self.train_batchsizes = torch.cat(
            [self.train_batchsizes, torch.tensor([x.shape[0]])]
        )
        self.train_total_weights += w.sum().item()

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        x, y, w = batch
        x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
        y_logits = self(x)
        loss = F.cross_entropy(y_logits, y, reduction="none") * w
        loss = loss.mean()

        self.val_losses = torch.cat([self.val_losses, loss.to("cpu").unsqueeze(0)])
        self.val_batchsizes = torch.cat(
            [self.val_batchsizes, torch.tensor([x.shape[0]])]
        )
        self.val_total_weights += w.sum().item()

        preds = torch.sigmoid(y_logits[:, 1] - y_logits[:, 0])

        self.val_preds = torch.cat((self.val_preds, preds.to("cpu")))
        self.val_labels = torch.cat((self.val_labels, y.to("cpu")))
        self.val_weights = torch.cat((self.val_weights, w.to("cpu")))

        return loss

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

    def on_train_epoch_start(self):
        self.log("step", self.trainer.current_epoch, on_epoch=True)

    def on_validation_epoch_start(self):
        self.log("step", self.trainer.current_epoch, on_epoch=True)

    def on_train_epoch_end(self):
        avg_loss = (
            torch.sum(self.train_losses * self.train_batchsizes)
            / self.train_total_weights
        )
        self.log("train_loss", avg_loss, on_epoch=True, prog_bar=True)

        self.train_losses = torch.tensor([])
        self.train_batchsizes = torch.tensor([])
        self.train_total_weights = 0.0

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
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("val_sigma_sq", avg_sigma_sq, on_epoch=True, prog_bar=True)

        last_lr = (
            self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[-1]
            if self.lr_scheduler_config["type"] != "none"
            else self.optimizer_config["lr"]
        )
        self.log("lr", last_lr, on_epoch=True)
        self.log("batch_size", self.datamodule.batch_size, on_epoch=True)

        self.val_losses = torch.tensor([])
        self.val_batchsizes = torch.tensor([])
        self.val_total_weights = 0.0
        self.val_preds = torch.tensor([])
        self.val_labels = torch.tensor([])
        self.val_weights = torch.tensor([])

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

        if train_seed is not None:
            pl.seed_everything(train_seed)

        torch.set_float32_matmul_precision("medium")

        tb_log_dir = pathlib.Path(f"./tb_logs/{tb_log_dir}")
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        logger = TensorBoardLogger(tb_log_dir, name=self.run_name)

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
    def predict(self, q: torch.Tensor, do_tqdm: bool = False) -> torch.Tensor:
        self.eval()
        batch_size = min(2**14, q.shape[0])
        q_dataloader = DataLoader(q, batch_size=batch_size, shuffle=False)

        preds = []
        if do_tqdm:
            q_dataloader = tqdm.tqdm(q_dataloader)
        for q_batch in q_dataloader:
            q_batch = q_batch.to(self.device)
            logit = self(q_batch)
            pred = F.softmax(logit, dim=-1)
            preds.append(pred.cpu())
            torch.cuda.empty_cache()  # Clear CUDA cache after each batch

        return torch.cat(preds, dim=0)
