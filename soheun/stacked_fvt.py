import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from data_modules import StackedFvTDataModule
from fvt_classifier import FvTClassifier
from utils import require_keys
import torch.optim as optim
from torch.utils.data import TensorDataset
import pathlib
import os
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pl_loggers import FileHandlerLogger
import numpy as np


class StackedFvTClassifier(pl.LightningModule):
    """
    Stacked FvT classifier. The interpretation of "stacking" depends on how
    the data is fed during training (e.g., same input, different heads/targets,
    or different inputs entirely).

    This implementation assumes each stack acts independently on the same input batch,
    and the loss is averaged across stacks unless y/w have a stack dimension.

    Args:
        num_stacks: number of independent FvT classifiers in the stack.
        num_classes: number of output classes for each classifier.
        dim_input_jet_features: input jet feature dimension.
        dim_dijet_features: dijet feature dimension.
        dim_quadjet_features: quadjet feature dimension.
    """

    def __init__(
        self,
        num_stacks: int,
        num_classes: int,
        dim_input_jet_features: int,
        dim_dijet_features: int,
        dim_quadjet_features: int,
        run_name: str,
        device: str = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        depth: dict = {
            "encoder": 4,
            "decoder": 1,
        },
        repr_norm: bool = False,
        # Add any other necessary hyperparameters
    ):
        super().__init__()
        # Saves all hyperparameters passed to __init__ (num_stacks, lr, etc.)
        # to self.hparams. Also allows accessing them directly like self.lr
        self.save_hyperparameters()

        self.num_stacks = num_stacks
        self.dim_j = dim_input_jet_features
        self.dim_d = dim_dijet_features
        self.dim_q = dim_quadjet_features

        self.num_classes = num_classes
        self.run_name = run_name

        require_keys(depth, ["encoder", "decoder"])
        self.depth = depth

        self.repr_norm = repr_norm

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
        # stacking multiple FvTClassifier module
        # Note: Ensure FvTClassifier's __init__ matches these args
        self.fvt_classifiers = nn.ModuleList(
            [
                FvTClassifier(
                    num_classes=self.num_classes,
                    dim_input_jet_features=self.dim_j,
                    dim_dijet_features=self.dim_d,
                    dim_quadjet_features=self.dim_q,
                    run_name=self.run_name,
                    device=self.device,
                    depth=self.depth,
                    repr_norm=self.repr_norm,
                )
                for _ in range(self.num_stacks)
            ]
        )

        # Use CrossEntropyLoss like the original FvTClassifier example
        # Assumes integer class labels for y
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through all stacked classifiers.

        Args:
            x: float tensor of shape (batch_size, num_stacks, input_features)
        Returns:
            logits_stack: float tensor of shape (batch_size, num_stacks, num_classes)
        """

        all_logits = [fvt(x[:, i, :]) for i, fvt in enumerate(self.fvt_classifiers)]
        logits_stack = torch.stack(all_logits, dim=1)
        return logits_stack

    def _calculate_loss(
        self, logits_stack: torch.Tensor, y: torch.Tensor, w: torch.Tensor
    ):
        """Helper to calculate weighted loss across stacks."""
        # logits_stack: (batch_size, num_stacks, num_classes)
        # y: (batch_size, num_stacks) - assumed integer labels
        # w: (batch_size, num_stacks)

        # Calculate loss for each stack element vs the target y
        # The CrossEntropyLoss expects logits as (N, C) and target as (N)
        # We need to calculate it for each stack item
        batch_size = y.size(0)
        assert logits_stack.shape == (
            batch_size,
            self.num_stacks,
            self.num_classes,
        ), f"logits_stack.shape: {logits_stack.shape}, expected: {(batch_size, self.num_stacks, self.num_classes)}"
        losses_per_stack = []
        for i in range(self.num_stacks):
            # logits for stack i: (batch_size, num_classes)
            stack_logits = logits_stack[:, i, :]
            # targets for stack i: (batch_size)
            stack_y = y[:, i]
            # weights for stack i: (batch_size)
            stack_w = w[:, i]
            # Calculate unreduced loss: (batch_size)
            # Pass the correct 1D target tensor (stack_y) for the current stack
            unreduced_loss = self.ce_loss(stack_logits, stack_y)
            # take mean across batch, not stack
            # Normalize by sum of weights to get weighted mean loss for this stack
            weighted_loss_mean = (unreduced_loss * stack_w).mean()
            losses_per_stack.append(weighted_loss_mean)

        # Sum the loss across all stacks (each loss is already batch-averaged)
        # Use torch.stack to preserve computation graph before summing
        final_loss = torch.stack(losses_per_stack).sum()
        # final_loss = torch.stack(losses_per_stack).mean() # Alternative: Average loss across stacks
        return final_loss

    def training_step(self, batch, batch_idx):
        x, y, w = batch  # Assumes standard batch structure
        x, y, w = (
            x.to(self.device),
            y.to(self.device),
            w.to(self.device),
        )  # Optional explicit transfer

        logits_stack = self(x)  # Shape: (batch_size, num_stacks, num_classes)
        loss = self._calculate_loss(logits_stack, y, w)

        # Log training loss
        self.log(
            "avg_train_loss",
            loss / self.num_stacks,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Optionally log learning rate
        # lr = self.optimizers().param_groups[0]['lr']
        # self.log('learning_rate', lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)

        logits_stack = self(x)
        loss = self._calculate_loss(logits_stack, y, w)

        # Log validation loss (accumulates over epoch)
        self.log(
            "avg_val_loss",
            loss / self.num_stacks,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

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
        self.log("epoch", self.trainer.current_epoch, on_epoch=True)
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
            "lr": last_lr,
            "batch_size": self.datamodule.batch_size,
        }

        self.update_history(tb_logs)

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        self.val_losses = torch.tensor([])
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

    def nan_check(self):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print("NaN found in parameter:", name)
                raise ValueError(f"NaN found in parameter: {name}")

    def test_step(self, batch, batch_idx):
        x, y, w = batch
        x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)

        logits_stack = self(x)
        loss = self._calculate_loss(logits_stack, y, w)

        # Log test loss (accumulates over epoch)
        self.log(
            "avg_test_loss",
            loss / self.num_stacks,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        # Log metrics similar to validation_step
        # preds_stack = torch.argmax(logits_stack, dim=-1)
        # acc_per_stack = (preds_stack == y.unsqueeze(0)).float().mean(dim=1)
        # avg_acc = acc_per_stack.mean()
        # self.log("test_acc", avg_acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return loss

    @torch.no_grad()
    def predict(self, x):
        """Defines prediction logic. Returns raw logits by default."""
        x = x.to(self.device)  # Optional explicit transfer
        logits_stack = self(x)
        return logits_stack

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
        train_datasets: list[TensorDataset],
        val_datasets: list[TensorDataset],
        max_epochs=50,
        train_seed: int = None,
        save_checkpoint: bool = True,
        callbacks: list[pl.Callback] = [],
        tb_log_dir: str = "tmp",
        optimizer_config: dict = {},
        lr_scheduler_config: dict = {},
        early_stop_patience: None | int = None,
        dataloader_config: dict = {},
        file_handler: logging.FileHandler | None = None,
        progress_bar_epochs: int = None,
    ):
        assert "batch_size" in dataloader_config

        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.early_stop_patience = early_stop_patience

        self.train_datasets = train_datasets
        self.val_datasets = val_datasets

        self.dataloader_config = dataloader_config

        if train_seed is not None:
            pl.seed_everything(train_seed)

        torch.set_float32_matmul_precision("medium")

        tb_log_dir = pathlib.Path(f"./tb_logs/{tb_log_dir}")
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        loggers = [TensorBoardLogger(tb_log_dir, name=self.run_name)]
        if file_handler is not None:
            file_logger = FileHandlerLogger(file_handler)
            loggers.append(file_logger)

        progress_bar = TQDMProgressBar(
            # refresh_rate=(
            #     max(1, (len(train_datasets[0]) // dataloader_config["batch_size"]) // 4)
            #     if progress_bar_epochs is None
            #     else progress_bar_epochs
            # )
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
            logger=loggers,
            reload_dataloaders_every_n_epochs=1,
            enable_progress_bar=True,
        )

        torch.autograd.set_detect_anomaly(True)

        num_workers = dataloader_config.get("num_workers", 0)
        pin_mem = dataloader_config.get("pin_memory", True)
        persist = dataloader_config.get("persistent_workers", True)
        self.datamodule = StackedFvTDataModule(
            train_datasets,
            val_datasets,
            dataloader_config["batch_size"],
            num_workers=num_workers,
            batch_size_milestones=dataloader_config.get("batch_size_milestones", []),
            batch_size_multiplier=dataloader_config.get("batch_size_multiplier", 2),
            pin_memory=pin_mem,
            persistent_workers=persist,
        )
        # Launch training
        trainer.fit(self, datamodule=self.datamodule)
