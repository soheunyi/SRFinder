"""
ensemble_classifiers.py

Provides LightningModule wrappers to train multiple copies of classifiers in parallel within a single process/GPU.
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import ModuleList
from torch.optim import Adam, SGD
from data_modules import FvTDataModule
from fvt_encoder import FvTEncoder
from network_blocks import ResNetBlock
from utils import require_keys


class MultiAttentionClassifier(pl.LightningModule):
    """
    Train N copies of the attention-based classifier in parallel.
    Forward returns a tensor of shape (n_models, batch_size, num_classes).
    """
    def __init__(
        self,
        dim_q: int,
        num_classes: int,
        depth: int = 1,
        n_models: int = 2,
        optimizer_config: dict | None = None,
        lr_scheduler_config: dict | None = None,
    ):
        super().__init__()
        self.dim_q = dim_q
        self.num_classes = num_classes
        self.depth = depth
        self.n_models = n_models
        # default optimizer and scheduler if not provided
        self.optimizer_config = optimizer_config or {"type": "Adam", "lr": 1e-3}
        self.lr_scheduler_config = lr_scheduler_config or {"type": "none"}

        # build separate blocks for each model
        self.select_q_blocks = ModuleList([
            ResNetBlock(dim_q, 1, depth) for _ in range(n_models)
        ])
        self.out_blocks = ModuleList([
            ResNetBlock(dim_q, num_classes, depth) for _ in range(n_models)
        ])

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: (batch_size, dim_q)
        logits = []
        for i in range(self.n_models):
            # compute attention-weighted event representation
            q_score = self.select_q_blocks[i](q)
            q_score = F.softmax(q_score, dim=-1)
            event = torch.matmul(q, q_score.transpose(1, 2)).view(q.size(0), self.dim_q, 1)
            logit = self.out_blocks[i](event).view(q.size(0), self.num_classes)
            logits.append(logit)
        # stack: (n_models, batch_size, num_classes)
        return torch.stack(logits, dim=0)

    def training_step(self, batch, batch_idx):  # type: ignore
        q, y, w = batch
        q, y, w = q.to(self.device), y.to(self.device), w.to(self.device)
        logits = self(q)  # (n_models, batch_size, num_classes)
        # reshape to compute per-model loss
        bs = y.size(0)
        # flatten for cross entropy: (n_models*batch_size, num_classes)
        logits_flat = logits.permute(1, 0, 2).reshape(-1, self.num_classes)
        y_rep = y.repeat(self.n_models)
        w_rep = w.repeat(self.n_models)
        loss_flat = F.cross_entropy(logits_flat, y_rep, reduction="none") * w_rep
        # reshape back: (n_models, batch_size)
        loss_per_model = loss_flat.view(self.n_models, bs)
        # mean over batch then models
        loss = loss_per_model.mean(dim=1).mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore
        q, y, w = batch
        q, y, w = q.to(self.device), y.to(self.device), w.to(self.device)
        logits = self(q)
        bs = y.size(0)
        logits_flat = logits.permute(1, 0, 2).reshape(-1, self.num_classes)
        y_rep = y.repeat(self.n_models)
        w_rep = w.repeat(self.n_models)
        loss_flat = F.cross_entropy(logits_flat, y_rep, reduction="none") * w_rep
        loss_per_model = loss_flat.view(self.n_models, bs)
        loss = loss_per_model.mean(dim=1).mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):  # type: ignore
        require_keys(self.optimizer_config, ["type", "lr"])
        typ = self.optimizer_config["type"]
        lr = self.optimizer_config["lr"]
        if typ == "Adam":
            optimizer = Adam(self.parameters(), lr=lr)
        elif typ == "SGD":
            optimizer = SGD(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Invalid optimizer type: {typ}")
        sched_conf = self.lr_scheduler_config
        if sched_conf["type"] == "none":
            return optimizer
        elif sched_conf["type"] == "ReduceLROnPlateau":
            require_keys(sched_conf, ["factor", "threshold", "patience", "cooldown", "min_lr"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                factor=sched_conf["factor"],
                threshold=sched_conf["threshold"],
                patience=sched_conf["patience"],
                cooldown=sched_conf["cooldown"],
                min_lr=sched_conf["min_lr"],
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        else:
            raise ValueError(f"Invalid lr scheduler type: {sched_conf['type']}")


from fvt_classifier import FvTClassifier
import copy

class MultiFvTClassifier(FvTClassifier):
    """
    Ensemble of N copies of FvTClassifier (encoder + attention) trained in parallel.
    Forward returns a tensor of shape (n_models, batch_size, num_classes).
    """
    def __init__(
        self,
        n_models: int,
        num_classes: int,
        dim_input_jet_features: int,
        dim_dijet_features: int,
        dim_quadjet_features: int,
        run_name: str,
        device=None,
        depth: dict = {"encoder": 4, "decoder": 1},
        repr_norm: bool = False,
    ):
        # Initialize single-model attributes via parent
        super().__init__(
            num_classes,
            dim_input_jet_features,
            dim_dijet_features,
            dim_quadjet_features,
            run_name,
            device,
            depth,
            repr_norm,
        )
        self.n_models = n_models
        # Keep original single-copy modules
        orig_encoder = self.encoder
        orig_attn = self.attention_classifier
        # Remove single-copy attributes
        del self.encoder, self.attention_classifier
        # Create lists of encoders and attention-classifiers
        self.encoder_list = ModuleList([
            copy.deepcopy(orig_encoder) for _ in range(n_models)
        ])
        self.attention_list = ModuleList([
            copy.deepcopy(orig_attn) for _ in range(n_models)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, ...)
        logits = []
        for i in range(self.n_models):
            q = self.encoder_list[i](x)
            class_score = self.attention_list[i](q)
            logits.append(class_score)
        # Stack over model axis: (n_models, batch_size, num_classes)
        return torch.stack(logits, dim=0)

    def training_step(self, batch, batch_idx):  # type: ignore
        x, y, w = batch
        x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
        logits = self(x)  # (n_models, batch_size, num_classes)
        n, bs, c = logits.shape
        flat = logits.permute(1, 0, 2).reshape(-1, c)
        y_rep = y.repeat(n)
        w_rep = w.repeat(n)
        loss_flat = F.cross_entropy(flat, y_rep, reduction="none") * w_rep
        loss_per = loss_flat.view(n, bs)
        loss = loss_per.mean(dim=1).mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore
        x, y, w = batch
        x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
        logits = self(x)
        n, bs, c = logits.shape
        flat = logits.permute(1, 0, 2).reshape(-1, c)
        y_rep = y.repeat(n)
        w_rep = w.repeat(n)
        loss_flat = F.cross_entropy(flat, y_rep, reduction="none") * w_rep
        loss_per = loss_flat.view(n, bs)
        loss = loss_per.mean(dim=1).mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, ... input dims as required by encoder)
        logits = []
        for i in range(self.n_models):
            q = self.encoder_list[i](x)
            # attention part
            q_score = self.select_q_blocks[i](q)
            q_score = F.softmax(q_score, dim=-1)
            event = torch.matmul(q, q_score.transpose(1, 2)).view(q.size(0), self.dim_q, 1)
            logit = self.out_blocks[i](event).view(q.size(0), self.num_classes)
            logits.append(logit)
        return torch.stack(logits, dim=0)

    def training_step(self, batch, batch_idx):  # type: ignore
        x, y, w = batch
        x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
        logits = self(x)  # (n_models, batch_size, num_classes)
        bs = y.size(0)
        # flatten for CE
        logits_flat = logits.permute(1, 0, 2).reshape(-1, self.num_classes)
        y_rep = y.repeat(self.n_models)
        w_rep = w.repeat(self.n_models)
        loss_flat = F.cross_entropy(logits_flat, y_rep, reduction="none") * w_rep
        loss_per_model = loss_flat.view(self.n_models, bs)
        loss = loss_per_model.mean(dim=1).mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore
        x, y, w = batch
        x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
        logits = self(x)
        bs = y.size(0)
        logits_flat = logits.permute(1, 0, 2).reshape(-1, self.num_classes)
        y_rep = y.repeat(self.n_models)
        w_rep = w.repeat(self.n_models)
        loss_flat = F.cross_entropy(logits_flat, y_rep, reduction="none") * w_rep
        loss_per_model = loss_flat.view(self.n_models, bs)
        loss = loss_per_model.mean(dim=1).mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):  # type: ignore
        require_keys(self.optimizer_config, ["type", "lr"])
        typ = self.optimizer_config["type"]
        lr = self.optimizer_config["lr"]
        if typ == "Adam":
            optimizer = Adam(self.parameters(), lr=lr)
        elif typ == "SGD":
            optimizer = SGD(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Invalid optimizer type: {typ}")
        sched_conf = self.lr_scheduler_config
        if sched_conf["type"] == "none":
            return optimizer
        elif sched_conf["type"] == "ReduceLROnPlateau":
            require_keys(sched_conf, ["factor", "threshold", "patience", "cooldown", "min_lr"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                factor=sched_conf["factor"],
                threshold=sched_conf["threshold"],
                patience=sched_conf["patience"],
                cooldown=sched_conf["cooldown"],
                min_lr=sched_conf["min_lr"],
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        else:
            raise ValueError(f"Invalid lr scheduler type: {sched_conf['type']}")