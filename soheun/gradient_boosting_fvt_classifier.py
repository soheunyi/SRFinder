import torch
import tqdm
import pytorch_lightning as pl
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar, Callback
import pathlib
from fvt_classifier import FvTClassifier


class GradientBoostFvTClassifier(FvTClassifier):
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
        max_depth: int = 3,
    ):
        super().__init__(
            num_classes=num_classes,
            dim_input_jet_features=dim_input_jet_features,
            dim_dijet_features=dim_dijet_features,
            dim_quadjet_features=dim_quadjet_features,
            run_name=run_name,
            device=device,
            lr=lr,
        )
        self.save_hyperparameters()

        assert max_depth >= 1

        self.dim_j = dim_input_jet_features
        self.dim_d = dim_dijet_features
        self.dim_q = dim_quadjet_features

        self.run_name = run_name
        self.max_depth = max_depth

        self.reset_fvt_classifiers()
        self.to(device)
        print("self.device", self.device)

        # remove self.encoder, self.select_q, self.out
        del self.encoder
        del self.select_q
        del self.out

        self.coefficients = nn.Parameter(
            torch.tensor([1.0] + [0.0] * (self.max_depth - 1))
        )
        self.coefficients.requires_grad = False

        self.train_losses = torch.tensor([])
        self.train_batchsizes = torch.tensor([])
        self.train_total_weights = 0.0
        self.validation_losses = torch.tensor([])
        self.validation_batchsizes = torch.tensor([])
        self.validation_total_weights = 0.0
        self.best_val_loss = torch.inf

    def reset_fvt_classifiers(self):
        self.fvt_classifiers = nn.ModuleList([])
        self.fvt_classifiers.append(self.new_fvt_classifier())

    def new_fvt_classifier(self):
        fvt_classifier = FvTClassifier(
            num_classes=self.num_classes,
            dim_input_jet_features=self.dim_j,
            dim_dijet_features=self.dim_d,
            dim_quadjet_features=self.dim_q,
            run_name="_".join([self.run_name, str(len(self.fvt_classifiers))]),
            device=self.device,
            lr=self.lr,
        )
        fvt_classifier = fvt_classifier.to(self.device)
        return fvt_classifier

    def forward(self, x: torch.Tensor, level: int = None):
        level = len(self.fvt_classifiers) - 1 if level is None else level

        n = x.shape[0]
        y_logits = torch.zeros(n, self.num_classes).to(self.device)
        for i, fvt_classifier in enumerate(self.fvt_classifiers):
            if i > level:
                break
            if self.coefficients[i] == 0:
                continue
            # if i == 0:
            #     y_logits += self.coefficients[i] * fvt_classifier(x)
            # else:
            #     y_logits += self.coefficients[i] * torch.sigmoid(fvt_classifier(x))
            y_logits += self.coefficients[i] * fvt_classifier(x)
        return y_logits

    def step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        x, y, w = batch  # x: features, y: labels, w: weights
        x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)

        if len(self.fvt_classifiers) == 1:
            # For the first classifier, use standard FvTClassifier training
            y_logits = self.fvt_classifiers[0](x)
            loss = self.loss(y_logits, y, reduction="none")

        else:
            # For subsequent classifiers, use gradient boosting
            with torch.no_grad():
                ensemble_logits = self(x)

            ensemble_probs = F.softmax(ensemble_logits, dim=-1)[:, 1]
            gradients = y - ensemble_probs

            current_classifier = self.fvt_classifiers[-1]
            current_logits = current_classifier(x)
            loss = F.mse_loss(
                current_logits[:, 1] - current_logits[:, 0], gradients, reduction="none"
            )
            # # # Use binary cross-entropy loss as an alternative
            # current_probs = F.softmax(current_logits, dim=-1)[:, 1]
            # # loss = F.binary_cross_entropy(
            # #     current_probs, 0.5 * (gradients + 1), reduction="none"
            # # )

        loss = (loss * w).mean()
        batch_size = x.shape[0]

        return loss, batch_size, w.sum()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx
    ):
        loss, batch_size, total_weight = self.step(batch)
        self.train_losses = torch.cat(
            (self.train_losses, loss.detach().to("cpu").view(1))
        )
        self.train_batchsizes = torch.cat(
            (self.train_batchsizes, torch.tensor(batch_size).view(1))
        )
        self.train_total_weights += total_weight

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx
    ):
        loss, batch_size, total_weight = self.step(batch)
        self.validation_losses = torch.cat(
            (self.validation_losses, loss.detach().to("cpu").view(1))
        )
        self.validation_batchsizes = torch.cat(
            (self.validation_batchsizes, torch.tensor(batch_size).view(1))
        )
        self.validation_total_weights += total_weight

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
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        self.validation_losses = torch.tensor([])
        self.validation_batchsizes = torch.tensor([])
        self.validation_total_weights = 0.0

        self.nan_check()

    def optimize_last_coefficient(self, val_dataset, batch_size):
        self.eval()
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        last_fvt_idx = len(self.fvt_classifiers) - 1

        def loss_fn(coeff):
            self.coefficients.data[last_fvt_idx] = torch.tensor(
                coeff, device=self.device
            )
            total_loss = 0
            total_samples = 0

            with torch.no_grad():
                for x, y, w in tqdm.tqdm(val_loader):
                    x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
                    y_logits = self(x)
                    loss = self.loss(y_logits, y, reduction="none")
                    loss = (loss * w).sum()
                    total_loss += loss.item()
                    total_samples += w.sum().item()
            return total_loss / total_samples

        grid = torch.tensor([0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]).to(self.device)
        losses = [loss_fn(coeff) for coeff in grid]
        best_idx = torch.argmin(torch.tensor(losses))
        optimized_coeff = grid[best_idx].item()

        self.coefficients.data[last_fvt_idx] = torch.tensor(
            optimized_coeff, device=self.device
        )
        print(f"Optimized coefficient: {optimized_coeff:.6f}")

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
        self.reset_fvt_classifiers()

        if train_seed is not None:
            pl.seed_everything(train_seed)

        torch.set_float32_matmul_precision("medium")

        logger = TensorBoardLogger("tb_logs", name=self.run_name)

        progress_bar = TQDMProgressBar(
            refresh_rate=max(1, (len(train_dataset) // batch_size) // 10)
        )
        callbacks = callbacks + [progress_bar]

        if save_checkpoint:
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

        torch.autograd.set_detect_anomaly(True)

        for fvt_idx in range(self.max_depth):
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=5,
                verbose=False,
                mode="min",
            )
            print(f"Training FvT Classifier {fvt_idx}")
            self.train()

            trainer = pl.Trainer(
                max_epochs=max_epochs,
                callbacks=callbacks + [early_stop_callback],
                logger=logger,
                accelerator="gpu",
                devices=1,
            )

            trainer.fit(
                self,
                DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
                ),
                DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
                ),
            )

            self.to("cuda")
            print("self.device", self.device)

            if fvt_idx > 0:
                print(f"Optimizing coefficient for Classifier {fvt_idx}")
                self.optimize_last_coefficient(val_dataset, batch_size)

            if fvt_idx + 1 < self.max_depth:
                print(f"Adding Gradient Boost Classifier {fvt_idx + 1}")
                self.fvt_classifiers.append(self.new_fvt_classifier())
