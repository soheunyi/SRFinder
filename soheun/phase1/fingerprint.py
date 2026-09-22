"""Non-invasive fingerprinting of the existing stacked CR training loop.

Nothing here changes training behaviour.  The callback only observes the
current ``StackedFvTClassifier`` run through Lightning hooks and records a
deterministic fingerprint:

* per-estimator initial and final parameter digests,
* per-epoch minibatch digests (captures shuffle order, batch order and the
  batch-size schedule),
* per-epoch per-estimator validation losses, learning rates and batch size,
* the ``callback_metrics`` snapshot that ``SaveIndividualClassifierCallback``
  actually consumes, which is one epoch stale because Lightning runs callbacks
  before the LightningModule hook that logs ``val_loss_stack_i``.  The
  per-stack ``ReduceLROnPlateau`` step is *not* affected: it runs in the
  module's ``on_train_epoch_end``, which fires after the validation loop, so
  it reads the current epoch's value.  Verified on lightning 2.2.1,
* per-estimator best score and best epoch as selected by the existing saver.

Phase 1 of the stacked-training issue requires two runs of the same
configuration to produce equal fingerprints before any later phase is allowed
to touch training behaviour.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytorch_lightning as pl
import torch


def tensor_digest(t: torch.Tensor) -> str:
    """sha256 over the exact bytes of a tensor, device independent."""
    a = t.detach().to("cpu").contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(a.shape)).encode())
    h.update(str(a.dtype).encode())
    h.update(a.numpy().tobytes())
    return h.hexdigest()


def module_digest(module: torch.nn.Module) -> str:
    """sha256 over a module's state_dict, in sorted key order."""
    h = hashlib.sha256()
    sd = module.state_dict()
    for key in sorted(sd.keys()):
        v = sd[key]
        h.update(key.encode())
        if isinstance(v, torch.Tensor):
            a = v.detach().to("cpu").contiguous()
            h.update(str(tuple(a.shape)).encode())
            h.update(str(a.dtype).encode())
            h.update(a.numpy().tobytes())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()


def stack_param_digests(pl_module) -> list[str]:
    return [module_digest(m) for m in pl_module.fvt_classifiers]


def env_record() -> dict:
    rec = {
        "torch": torch.__version__,
        "lightning": pl.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if torch.cuda.is_available():
        rec["cuda"] = torch.version.cuda
        rec["cudnn"] = torch.backends.cudnn.version()
        rec["gpu_name"] = torch.cuda.get_device_name(0)
        rec["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    return rec


class FingerprintCallback(pl.Callback):
    """Observe-only callback.  Must be registered before the saver callback."""

    def __init__(self, num_stacks: int, out_path: str | pathlib.Path):
        super().__init__()
        self.num_stacks = num_stacks
        self.out_path = pathlib.Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        self.record: dict = {
            "num_stacks": num_stacks,
            "env": {},
            "init_param_digests": None,
            "final_param_digests": None,
            "epochs": [],  # one entry per training epoch
            "val_epochs": [],  # one entry per validation epoch
            "best": None,
            "saver_best_scores": None,
        }

        # mirror of SaveIndividualClassifierCallback's selection rule
        self._best_scores = [float("inf")] * num_stacks
        self._best_epochs = [-1] * num_stacks

        self._epoch_batches: list[str] = []
        self._epoch_batch_sizes: list[int] = []
        self._epoch_running = hashlib.sha256()

    # ------------------------------------------------------------------ train

    def on_train_start(self, trainer, pl_module):
        self.record["env"] = env_record()
        if self.record["init_param_digests"] is None:
            self.record["init_param_digests"] = stack_param_digests(pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_batches = []
        self._epoch_batch_sizes = []
        self._epoch_running = hashlib.sha256()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        x, y, w = batch
        h = hashlib.sha256()
        for t in (x, y, w):
            h.update(tensor_digest(t).encode())
        d = h.hexdigest()
        self._epoch_batches.append(d)
        self._epoch_batch_sizes.append(int(y.shape[0]))
        self._epoch_running.update(d.encode())

    def on_train_epoch_end(self, trainer, pl_module):
        dm = getattr(pl_module, "datamodule", None)
        self.record["epochs"].append(
            {
                "epoch": int(trainer.current_epoch),
                "num_batches": len(self._epoch_batches),
                "batch_sizes": self._epoch_batch_sizes,
                "batch_order_digest": self._epoch_running.hexdigest(),
                "first_batch_digest": (
                    self._epoch_batches[0] if self._epoch_batches else None
                ),
                "dm_batch_size": int(dm.batch_size) if dm is not None else None,
                "lrs": [
                    float(opt.param_groups[0]["lr"]) for opt in trainer.optimizers
                ],
            }
        )

    # ------------------------------------------------------------------- val

    def _stale_metrics(self, trainer) -> list[float | None]:
        out = []
        for i in range(self.num_stacks):
            v = trainer.callback_metrics.get(f"val_loss_stack_{i}", None)
            if v is None:
                out.append(None)
            else:
                out.append(float(v.item() if hasattr(v, "item") else v))
        return out

    def on_validation_epoch_end(self, trainer, pl_module):
        """Runs BEFORE the LightningModule hook, i.e. the same stale view of
        ``callback_metrics`` that ``SaveIndividualClassifierCallback`` consumes.
        The LR scheduler steps later, in the module's ``on_train_epoch_end``,
        and does not see this stale value."""
        if trainer.sanity_checking:
            return
        stale = self._stale_metrics(trainer)
        self._pending_stale = stale
        for i, v in enumerate(stale):
            if v is not None and v < self._best_scores[i]:
                self._best_scores[i] = v
                self._best_epochs[i] = int(trainer.current_epoch)

    def on_validation_end(self, trainer, pl_module):
        """Runs after the LightningModule hook, so metrics are current."""
        if trainer.sanity_checking:
            return
        fresh = self._stale_metrics(trainer)
        dm = getattr(pl_module, "datamodule", None)
        self.record["val_epochs"].append(
            {
                "epoch": int(trainer.current_epoch),
                "val_loss_per_stack": fresh,
                "val_loss_per_stack_as_seen_by_saver": getattr(
                    self, "_pending_stale", None
                ),
                "lrs": [
                    float(opt.param_groups[0]["lr"]) for opt in trainer.optimizers
                ],
                "dm_batch_size": int(dm.batch_size) if dm is not None else None,
            }
        )

    # ------------------------------------------------------------------- end

    def on_fit_end(self, trainer, pl_module):
        self.record["final_param_digests"] = stack_param_digests(pl_module)
        self.record["best"] = {
            "scores": self._best_scores,
            "epochs": self._best_epochs,
        }
        for cb in trainer.callbacks:
            if type(cb).__name__.endswith("SaveIndividualClassifierCallback"):
                self.record["saver_best_scores"] = [
                    cb.best_scores[m] for m in cb.monitor_metrics
                ]
        self.dump()

    def dump(self):
        with open(self.out_path, "w") as f:
            json.dump(self.record, f, indent=2, sort_keys=True)
