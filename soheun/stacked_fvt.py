import datetime
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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pl_loggers import FileHandlerLogger
from pl_callbacks import SaveIndividualClassifierCallback
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
        run_names: list[str],
        stacked_run_name: str,
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
        self.run_names = run_names
        self.stacked_run_name = stacked_run_name
        require_keys(depth, ["encoder", "decoder"])
        self.depth = depth

        self.repr_norm = repr_norm

        self.optimizer_config = None
        self.lr_scheduler_config = None

        self.train_losses_per_stack = torch.tensor([], dtype=torch.float32).view(
            0, self.num_stacks
        )
        self.train_weights_per_stack = torch.tensor([], dtype=torch.float32).view(
            0, self.num_stacks
        )
        self.train_batch_sizes = torch.tensor([])

        self.val_losses_per_stack = torch.tensor([], dtype=torch.float32).view(
            0, self.num_stacks
        )
        self.val_weights_per_stack = torch.tensor([], dtype=torch.float32).view(
            0, self.num_stacks
        )
        self.val_batch_sizes = torch.tensor([])
        self.best_val_loss = torch.inf

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
                    run_name=self.run_names[i],
                    device=self.device,
                    depth=self.depth,
                    repr_norm=self.repr_norm,
                )
                for i in range(self.num_stacks)
            ]
        )

        self.to(device)

        # Disable automatic optimization for learning rate scheduling per stack
        self.automatic_optimization = False

    def ce_loss(self, y_logits: torch.Tensor, y_labels: torch.Tensor, reduction="none"):
        # y_pred: logits, y_labels: labels
        return F.cross_entropy(y_logits, y_labels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through all stacked classifiers.

        Args:
            x: float tensor of shape (batch_size, num_stacks, input_features)
        Returns:
            logits_stack: float tensor of shape (batch_size, num_stacks, num_classes)
        """

        # Standard approach - process each model sequentially with its complete batch
        all_logits = []
        for i, fvt in enumerate(self.fvt_classifiers):
            # Extract this stack's input for all batch items
            stack_input = x[:, i, :]
            # Process the entire batch in one go, ensuring gradient flow
            logits = fvt(stack_input)
            all_logits.append(logits)

        # Stack results along stack dimension
        logits_stack = torch.stack(all_logits, dim=1)
        return logits_stack

    def _calculate_losses(
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
        return torch.stack(losses_per_stack)

    def training_step(self, batch, batch_idx):
        optimizers: list[torch.optim.Optimizer] = self.optimizers()
        x, y, w = batch  # Assumes standard batch structure
        x, y, w = (
            x.to(self.device),
            y.to(self.device),
            w.to(self.device),
        )
        self.train_batch_sizes = torch.cat(
            (self.train_batch_sizes, torch.tensor([y.size(0)]))
        )

        # Track individual losses for logging
        batch_losses = []
        batch_weights = []

        # Process each classifier independently with its own forward/backward pass
        for i, fvt in enumerate(self.fvt_classifiers):
            optimizers[i].zero_grad()

            # Extract this stack's input and targets
            stack_input = x[:, i, :]
            stack_y = y[:, i]
            stack_w = w[:, i]

            # Forward pass for this stack only
            logits = fvt(stack_input)

            # Calculate loss for this stack
            unreduced_loss = self.ce_loss(logits, stack_y)
            loss = (unreduced_loss * stack_w).mean()

            # Backward pass and optimizer step
            self.manual_backward(loss)
            optimizers[i].step()

            batch_losses.append(loss.detach())
            batch_weights.append(stack_w.sum())

        # Stack losses for logging
        stacked_losses = torch.stack(batch_losses)
        stacked_weights = torch.stack(batch_weights)

        self.train_losses_per_stack = torch.cat(
            (
                self.train_losses_per_stack,
                stacked_losses.detach().to("cpu").view(1, -1),
            ),
            dim=0,
        )
        self.train_weights_per_stack = torch.cat(
            (
                self.train_weights_per_stack,
                stacked_weights.detach().to("cpu").view(1, -1),
            ),
            dim=0,
        )

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
        self.val_batch_sizes = torch.cat(
            (self.val_batch_sizes, torch.tensor([y.size(0)]))
        )

        # Calculate individual losses for each stack
        batch_losses = []
        batch_weights = []

        for i, fvt in enumerate(self.fvt_classifiers):
            # Extract this stack's input and targets
            stack_input = x[:, i, :]
            stack_y = y[:, i]
            stack_w = w[:, i]

            # Forward pass for this stack only
            logits = fvt(stack_input)

            # Calculate loss for this stack
            unreduced_loss = self.ce_loss(logits, stack_y)
            loss = (unreduced_loss * stack_w).mean()
            batch_losses.append(loss)

            # Store the sum of weights for this stack
            batch_weights.append(stack_w.sum())

        # Stack losses and weights for all classifiers in this batch
        stacked_losses = torch.stack(batch_losses)  # [num_stacks]
        stacked_weights = torch.stack(batch_weights)  # [num_stacks]

        # Add to our running validation metrics
        self.val_losses_per_stack = torch.cat(
            (self.val_losses_per_stack, stacked_losses.detach().to("cpu").view(1, -1)),
            dim=0,
        )  # [number of batches, num_stacks]

        self.val_weights_per_stack = torch.cat(
            (
                self.val_weights_per_stack,
                stacked_weights.detach().to("cpu").view(1, -1),
            ),
            dim=0,
        )  # [number of batches, num_stacks]

    def on_train_epoch_end(self):
        avg_losses = torch.sum(
            self.train_losses_per_stack * self.train_batch_sizes.view(-1, 1), dim=0
        ) / torch.sum(self.train_weights_per_stack, dim=0)
        avg_loss = torch.mean(avg_losses)
        avg_loss_first_digits = (
            int(avg_loss.item() * 1000) if not torch.isnan(avg_loss) else 0
        )
        avg_loss_second_digits = (
            avg_loss.item() * 1000 - int(avg_loss.item() * 1000)
            if not torch.isnan(avg_loss)
            else 0
        )
        self.log("epoch", self.trainer.current_epoch, on_epoch=True)
        self.log("avg_train_loss", avg_loss, on_epoch=True)
        self.log(
            "1000x_avg_train_loss_lower_digits",
            avg_loss_first_digits,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "1000x_avg_train_loss_second_digits",
            avg_loss_second_digits,
            on_epoch=True,
            prog_bar=True,
        )

        self.nan_check()

        # Correctly iterate through scheduler configurations
        scheduler_configs = self.trainer.lr_scheduler_configs
        if scheduler_configs:  # Check if list is not empty
            assert len(scheduler_configs) == self.num_stacks
            for i, config in enumerate(scheduler_configs):
                scheduler = config.scheduler
                monitor = f"val_loss_stack_{i}"
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if monitor not in self.trainer.callback_metrics:
                        logging.warning(
                            f"Metric '{monitor}' not found in callback_metrics at epoch {self.current_epoch}. Skipping ReduceLROnPlateau step."
                        )
                        continue
                    metric_value = self.trainer.callback_metrics[monitor]
                    scheduler.step(metric_value)
                else:
                    scheduler.step()

        self.train_losses_per_stack = torch.tensor([]).view(0, self.num_stacks)
        self.train_weights_per_stack = torch.tensor([]).view(0, self.num_stacks)
        self.train_batch_sizes = torch.tensor([])

    def on_validation_epoch_end(self):
        avg_losses = torch.sum(
            self.val_losses_per_stack * self.val_batch_sizes.view(-1, 1), dim=0
        ) / torch.sum(self.val_weights_per_stack, dim=0)
        avg_loss = torch.mean(avg_losses)
        avg_loss_first_digits = (
            int(avg_loss.item() * 1000) if not torch.isnan(avg_loss) else 0
        )
        avg_loss_second_digits = (
            avg_loss.item() * 1000 - int(avg_loss.item() * 1000)
            if not torch.isnan(avg_loss)
            else 0
        )

        # Update to handle multiple schedulers
        if self.lr_scheduler_config["type"] != "none":
            # You might want to log the LR of the first optimizer or average across all
            last_lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[-1]
        else:
            last_lr = self.optimizer_config["lr"]

        for i in range(self.num_stacks):
            self.log(
                f"val_loss_stack_{i}",
                avg_losses[i].item(),
                on_epoch=True,
                prog_bar=False,
            )
        self.log(
            "1000x_avg_val_loss_first_digits",
            avg_loss_first_digits,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "1000x_avg_val_loss_second_digits",
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

        self.val_losses_per_stack = torch.tensor([]).view(0, self.num_stacks)
        self.val_weights_per_stack = torch.tensor([]).view(0, self.num_stacks)
        self.val_batch_sizes = torch.tensor([])

        self.nan_check()

    def on_validation_epoch_start(self):
        print(
            "Current time:",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            flush=True,
        )
        print(self.trainer.lr_scheduler_configs, flush=True)

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

        # Create separate optimizers for each classifier
        optimizers = []
        schedulers = []

        for i, fvt in enumerate(self.fvt_classifiers):
            # You can customize optimizer settings per fvt if needed
            if self.optimizer_config["type"] == "Adam":
                # Can customize lr or other parameters per fvt
                opt = optim.Adam(fvt.parameters(), lr=self.optimizer_config["lr"])
            elif self.optimizer_config["type"] == "SGD":
                opt = optim.SGD(fvt.parameters(), lr=self.optimizer_config["lr"])
            else:
                raise ValueError(
                    f"Invalid optimizer type: {self.optimizer_config['type']}"
                )

            optimizers.append(opt)

            # Create scheduler for each optimizer if needed
            if self.lr_scheduler_config["type"] != "none":
                if self.lr_scheduler_config["type"] == "ReduceLROnPlateau":
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        opt,
                        "min",
                        factor=self.lr_scheduler_config["factor"],
                        threshold=self.lr_scheduler_config["threshold"],
                        patience=self.lr_scheduler_config["patience"],
                        cooldown=self.lr_scheduler_config["cooldown"],
                        min_lr=self.lr_scheduler_config["min_lr"],
                    )
                    # Monitor the specific loss for this stack are set in on_train_epoch_end
                    schedulers.append(scheduler)

        # Return in the format expected by PyTorch Lightning
        if self.lr_scheduler_config["type"] == "none":
            return optimizers
        else:
            return optimizers, schedulers

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
        loggers = [TensorBoardLogger(tb_log_dir, name=self.stacked_run_name)]
        if file_handler is not None:
            file_logger = FileHandlerLogger(file_handler)
            loggers.append(file_logger)

        progress_bar = TQDMProgressBar(
            refresh_rate=(
                max(
                    1, (len(train_datasets[0]) // dataloader_config["batch_size"]) // 32
                )
                if progress_bar_epochs is None
                else progress_bar_epochs
            )
        )
        callbacks = callbacks + [progress_bar]

        if self.early_stop_patience is not None:
            raise NotImplementedError("Early stopping not implemented for stacked FvT")

        if save_checkpoint:
            checkpoint_dir = pathlib.Path(f"./data/checkpoints/")
        else:
            checkpoint_dir = pathlib.Path(f"./data/tmp/checkpoints/")

        callbacks.append(
            SaveIndividualClassifierCallback(
                save_dir=checkpoint_dir,
                run_names=self.run_names,
                monitor_metrics=[f"val_loss_stack_{i}" for i in range(self.num_stacks)],
            )
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            logger=loggers,
            reload_dataloaders_every_n_epochs=1,
            enable_progress_bar=True,
        )

        torch.autograd.set_detect_anomaly(True)

        num_workers = dataloader_config.get("num_workers", 0)
        pin_mem = dataloader_config.get("pin_memory", False)
        persist = dataloader_config.get("persistent_workers", False)
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
