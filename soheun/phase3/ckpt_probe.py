"""What does pytorch_lightning 2.2.1 actually persist in a checkpoint?

Phase 3 must guarantee the stacked checkpoint carries model params, optimizer
and scheduler state, epoch, RNG state, batch-size schedule state, per-estimator
best scores and callback state. This probe establishes which of those Lightning
gives for free and which have to be added, rather than assuming either way.
"""

import pathlib
import tempfile

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

CALLS = []


class StatefulCallback(pl.Callback):
    """Does Lightning call state_dict/load_state_dict on callbacks?"""

    def __init__(self):
        self.best = 123.0

    def state_dict(self):
        CALLS.append("callback.state_dict")
        return {"best": self.best}

    def load_state_dict(self, state_dict):
        CALLS.append(f"callback.load_state_dict -> {state_dict}")
        self.best = state_dict["best"]


class StatefulDM(pl.LightningDataModule):
    """Does Lightning call state_dict/load_state_dict on the datamodule?"""

    def __init__(self):
        super().__init__()
        self.batch_size = 4
        self.ds = TensorDataset(torch.randn(16, 1))

    def state_dict(self):
        CALLS.append("datamodule.state_dict")
        return {"batch_size": self.batch_size}

    def load_state_dict(self, state_dict):
        CALLS.append(f"datamodule.load_state_dict -> {state_dict}")
        self.batch_size = state_dict["batch_size"]

    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size)


class M(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)

    def training_step(self, batch, _):
        (x,) = batch
        return self.lin(x).pow(2).mean()

    def validation_step(self, batch, _):
        (x,) = batch
        self.log("val_loss", self.lin(x).pow(2).mean())

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=1)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched,
                                                   "monitor": "val_loss"}}


tmp = pathlib.Path(tempfile.mkdtemp())
cb = StatefulCallback()
mc = ModelCheckpoint(dirpath=tmp, save_last=True, every_n_epochs=1)
dm = StatefulDM()

trainer = pl.Trainer(
    max_epochs=2,
    default_root_dir=tmp,
    callbacks=[cb, mc],
    logger=False,
    enable_progress_bar=False,
    enable_model_summary=False,
    num_sanity_val_steps=0,
)
trainer.fit(M(), datamodule=dm)

ckpt = torch.load(tmp / "last.ckpt", map_location="cpu")

print(f"\nlightning {pl.__version__}, torch {torch.__version__}")
print(f"\ncheckpoint top-level keys:")
for k in sorted(ckpt):
    v = ckpt[k]
    kind = type(v).__name__
    extra = ""
    if isinstance(v, dict):
        extra = f"  keys={sorted(v)[:6]}"
    elif isinstance(v, list):
        extra = f"  len={len(v)}"
    print(f"  {k:<28} {kind}{extra}")

print("\nwhat we need, and whether it is there:")
checks = {
    "model parameters": "state_dict" in ckpt,
    "optimizer states": bool(ckpt.get("optimizer_states")),
    "scheduler states": bool(ckpt.get("lr_schedulers")),
    "current epoch": "epoch" in ckpt,
    "global step": "global_step" in ckpt,
    "loop state": "loops" in ckpt,
    "callback state": bool(ckpt.get("callbacks")),
    "datamodule state": "datamodule" in ckpt or "datamodule_state_dict" in ckpt,
    "torch RNG": any("rng" in str(k).lower() for k in ckpt),
}
for name, present in checks.items():
    print(f"  {'YES' if present else 'NO ':<4} {name}")

print(f"\ncallbacks block: {list((ckpt.get('callbacks') or {}).keys())}")
print(f"\nstate_dict/load_state_dict calls observed: {CALLS}")

# now resume and see which load hooks fire
CALLS.clear()
cb2 = StatefulCallback()
cb2.best = -999.0  # will be overwritten if Lightning restores it
dm2 = StatefulDM()
dm2.batch_size = -1
trainer2 = pl.Trainer(
    max_epochs=3,
    default_root_dir=tmp,
    callbacks=[cb2, ModelCheckpoint(dirpath=tmp, save_last=True)],
    logger=False,
    enable_progress_bar=False,
    enable_model_summary=False,
    num_sanity_val_steps=0,
)
trainer2.fit(M(), datamodule=dm2, ckpt_path=str(tmp / "last.ckpt"))
print(f"\non resume, calls: {CALLS}")
print(f"  callback.best restored to {cb2.best} (123.0 means yes)")
print(f"  datamodule.batch_size restored to {dm2.batch_size} (4 means yes)")
