from typing import Literal
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from network_blocks import conv1d
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar

from fvt_classifier import FvTClassifier


class AttentionClassifier(pl.LightningModule):
    def __init__(
        self,
        input_size,
        num_classes,
        learning_rate=1e-3,
        random_state=42,
        fvt_hash=None,
        hidden_size=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.random_state = random_state

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        if fvt_hash is not None:
            fvt_model = FvTClassifier.load_from_checkpoint(
                f"./data/checkpoints/{fvt_hash}_best.ckpt"
            )
            fvt_model.eval()
            self.select_q = fvt_model.select_q
            self.out = fvt_model.out
        else:
            self.select_q = conv1d(
                input_size, 1, 1, name="quadjet selector", batchNorm=True
            )
            self.out = conv1d(input_size, num_classes, 1, name="out", batchNorm=True)

        self.val_losses = []
        self.val_weights = []

    def forward(self, q: torch.Tensor):
        n = q.shape[0]
        q_score = self.select_q(q)
        q_score = F.softmax(q_score, dim=-1)
        event = torch.matmul(q, q_score.transpose(1, 2))
        event = event.view(n, self.input_size, 1)

        # project the final event-level pixel into the class score space
        class_score = self.out(event)
        class_score = class_score.view(n, self.num_classes)

        return class_score

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        logits = self(x)
        loss = torch.sum(F.cross_entropy(logits, y) * w)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        logits = self(x)
        loss = torch.sum(F.cross_entropy(logits, y) * w)
        self.log("val_loss", loss)
        self.val_losses.append(loss)
        self.val_weights.append(w.sum())
        return loss

    def on_validation_epoch_end(self):
        val_losses = torch.tensor(self.val_losses)
        val_weights = torch.tensor(self.val_weights)
        avg_loss = torch.sum(val_losses) / torch.sum(val_weights)
        self.log("avg_val_loss", avg_loss, prog_bar=True)

        self.val_losses = []
        self.val_weights = []

        self.train()

    def test_step(self, batch, batch_idx):
        x, y, w = batch
        logits = self(x)
        loss = torch.sum(F.cross_entropy(logits, y) * w)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def predict(self, x) -> np.ndarray:
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        logits = self(x)
        return torch.softmax(logits, dim=1).detach().cpu().numpy()

    def fit(self, X, y, w, max_epochs=10):
        self.train()
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y, w, test_size=1 / 3, random_state=self.hparams.random_state
        )

        # fit batch size to the number of samples
        batch_size = 2**10

        X_train = X_train[: len(X_train) // batch_size * batch_size]
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = y_train[: len(y_train) // batch_size * batch_size]
        y_train = torch.tensor(y_train, dtype=torch.long)
        w_train = w_train[: len(w_train) // batch_size * batch_size]
        w_train = torch.tensor(w_train, dtype=torch.float32)

        X_val = X_val[: len(X_val) // batch_size * batch_size]
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = y_val[: len(y_val) // batch_size * batch_size]
        y_val = torch.tensor(y_val, dtype=torch.long)
        w_val = w_val[: len(w_val) // batch_size * batch_size]
        w_val = torch.tensor(w_val, dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train, w_train)
        val_dataset = TensorDataset(X_val, y_val, w_val)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        early_stop_callback = EarlyStopping(
            monitor="avg_val_loss", min_delta=0.0, patience=5, mode="min"
        )
        progress_bar = TQDMProgressBar(
            refresh_rate=max(1, (len(train_loader) // batch_size) // 10)
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stop_callback, progress_bar],
            logger=False,
            enable_model_summary=False,
            enable_checkpointing=False,
        )

        trainer.fit(self, train_loader, val_loader)


def smeared_density_ratio(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    noise_scale: float,
    base_noise_scale: Literal["minmax", "std"] = "minmax",
    pretrained_fvt_hash: str = None,
    max_epochs: int = 10,
    seed: int = 42,
):
    """
    Calculate the smeared density ratio.

    Parameters
    ----------
    X : np.ndarray
        Training data.
    y : np.ndarray
        Training labels.
    w : np.ndarray
        Training weights.
    noise_scale : float
        Scale of the noise.
    base_noise_scale : Literal["minmax", "std"], optional
        Base scale of the noise, by default "minmax".
    pretrained_fvt_hash : str, optional
        Hash of the pretrained FvT model, by default None.
    training : bool, optional
        Whether to train the model, by default True.
    """

    assert len(X) == len(y) == len(w)

    n = X.shape[0]
    n_classes = 2

    # add noise to the data
    base_noise_scale = (
        np.std(X, axis=0)
        if base_noise_scale == "std"
        else np.max(X, axis=0) - np.min(X, axis=0)
    )
    X = X + np.random.normal(0, noise_scale, X.shape) * base_noise_scale

    # create the model
    model = AttentionClassifier(
        input_size=X.shape[1],
        num_classes=n_classes,
        learning_rate=1e-3,
        random_state=seed,
        fvt_hash=pretrained_fvt_hash,
    )

    model.fit(X, y, w, max_epochs=max_epochs)
    model.eval()

    def density_ratio(x: np.ndarray) -> np.ndarray:
        prob_4b = model.predict(x)
        return prob_4b[:, 1] / prob_4b[:, 0]

    return density_ratio
