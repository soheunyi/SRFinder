"""Determine, for this exact Lightning version, whether the metric logged in
LightningModule.on_validation_epoch_end is visible to:

  (a) Callback.on_validation_epoch_end   <- SaveIndividualClassifierCallback
  (b) LightningModule.on_train_epoch_end <- the per-stack ReduceLROnPlateau step

Mirrors StackedFvTClassifier: the metric is logged in the module's
on_validation_epoch_end, and read from trainer.callback_metrics elsewhere.
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

ORDER = []


def seen(trainer):
    v = trainer.callback_metrics.get("val_loss_stack_0")
    return None if v is None else round(float(v), 4)


class Probe(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        ORDER.append(
            f"  ep{trainer.current_epoch} CALLBACK.on_validation_epoch_end   sees {seen(trainer)}"
        )

    def on_train_epoch_end(self, trainer, pl_module):
        ORDER.append(
            f"  ep{trainer.current_epoch} CALLBACK.on_train_epoch_end        sees {seen(trainer)}"
        )


class M(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.lin(x)

    def training_step(self, batch, _):
        (x,) = batch
        return self(x).pow(2).mean()

    def validation_step(self, batch, _):
        pass

    def on_validation_epoch_end(self):
        # epoch N logs the value N.0, so staleness is unmistakable
        val = float(self.current_epoch)
        self.log("val_loss_stack_0", val, on_epoch=True)
        ORDER.append(
            f"  ep{self.current_epoch} MODULE.on_validation_epoch_end     logs  {val}"
        )

    def on_train_epoch_end(self):
        ORDER.append(
            f"  ep{self.current_epoch} MODULE.on_train_epoch_end          sees {seen(self.trainer)}"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


ds = TensorDataset(torch.randn(8, 1))
trainer = pl.Trainer(
    max_epochs=4,
    enable_progress_bar=False,
    enable_model_summary=False,
    logger=False,
    enable_checkpointing=False,
    num_sanity_val_steps=0,
    callbacks=[Probe()],
)
trainer.fit(M(), DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4))

print(f"\nlightning {pl.__version__}\n")
print("hook order, with the value each site reads from callback_metrics:")
print("(epoch N logs the value N.0)\n")
print("\n".join(ORDER))
