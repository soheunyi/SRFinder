import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from fvt_encoder import FvTEncoder


class MultiTaskFvTClassifier(pl.LightningModule):
    """
    Multi-task classifier that shares a single FvTEncoder backbone
    and maintains separate linear heads for each task.

    Args:
        num_tasks: number of binary classification tasks.
        dim_input_jet_features: input jet feature dimension.
        dim_dijet_features: dijet feature dimension.
        dim_quadjet_features: quadjet feature dimension.
        lr: learning rate for the optimizer.
    """

    def __init__(
        self,
        num_tasks: int,
        dim_input_jet_features: int,
        dim_dijet_features: int,
        dim_quadjet_features: int,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Shared encoder
        self.encoder = FvTEncoder(
            dim_input_jet_features=dim_input_jet_features,
            dim_dijet_features=dim_dijet_features,
            dim_quadjet_features=dim_quadjet_features,
        )

        # Per-task linear classifiers
        dim_q = self.encoder.output_dim
        self.classifier_weight = nn.Parameter(torch.randn(num_tasks, dim_q))
        self.classifier_bias = nn.Parameter(torch.zeros(num_tasks))

        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float tensor of shape (batch_size, input_features)
        Returns:
            logits: float tensor of shape (batch_size, num_tasks)
        """
        # Encode inputs → (B, dim_q, symm)
        q = self.encoder(x)
        # Average over symmetry dimension → (B, dim_q)
        q = q.mean(dim=-1)
        # Linear projection for all tasks at once → (B, num_tasks)
        logits = F.linear(q, self.classifier_weight, self.classifier_bias)
        return logits

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        # x: (B, feat), y,w: (B, num_tasks)
        logits = self(x)
        loss_matrix = F.binary_cross_entropy_with_logits(
            logits, y.float(), reduction="none"
        )
        loss = (loss_matrix * w).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        logits = self(x)
        loss_matrix = F.binary_cross_entropy_with_logits(
            logits, y.float(), reduction="none"
        )
        loss = (loss_matrix * w).mean()
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# Example usage:
# from torch.utils.data import DataLoader, TensorDataset
# X: Tensor (N, feat_dim)
# Ys: list of num_tasks tensors (N,) in {0,1}
# Ws: list of num_tasks weight tensors (N,)
# Y = torch.stack(Ys, dim=1)  # shape (N, num_tasks)
# W = torch.stack(Ws, dim=1)  # shape (N, num_tasks)
# ds = TensorDataset(X, Y, W)
# train_loader = DataLoader(ds, batch_size=1024, shuffle=True, num_workers=8, pin_memory=True)
# val_loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=8, pin_memory=True)
# model = MultiTaskFvTClassifier(num_tasks=Y.shape[1],
#                                 dim_input_jet_features=..., dim_dijet_features=..., dim_quadjet_features=...)
# trainer = pl.Trainer(gpus=1, max_epochs=10)
# trainer.fit(model, train_loader, val_loader)
