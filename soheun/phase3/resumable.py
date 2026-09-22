"""Phase 3: make stacked CR training resumable, on Lightning's own machinery.

A probe of pytorch_lightning 2.2.1 (``phase3/ckpt_probe.py``) establishes what a
checkpoint already carries and what it does not:

===========================  =========================================
model parameters             free
optimizer states             free
scheduler states             free
epoch / global_step / loops  free
callback state               free, if the callback defines state_dict
datamodule state             free, if the datamodule defines state_dict
RNG states                   **not saved** -- must be added
===========================  =========================================

So Phase 3 is mostly a matter of using what is already there. Today
``StackedFvTClassifier.fit`` registers no ``ModelCheckpoint`` at all, which is
why there is no ``last.ckpt`` to resume from; the individual ``*_best.pt`` files
it does write hold weights only.

This module supplies four pieces and a patch installer. None of them edits a
production file; they are substituted into the ``stacked_fvt`` module namespace
for the duration of a run, the same way Phases 1 and 2 do it, so every line of
``fit`` still executes as written.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import random
from typing import Any, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.io import TorchCheckpointIO

from data_modules import StackedFvTDataModule
from pl_callbacks import SaveIndividualClassifierCallback


# --------------------------------------------------------------- atomic writes


def atomic_torch_save(obj: Any, path: str | pathlib.Path) -> None:
    """Write via a temporary file and rename, so a kill mid-write cannot leave
    a truncated checkpoint behind. ``os.replace`` is atomic within a
    filesystem."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class AtomicCheckpointIO(TorchCheckpointIO):
    """``CheckpointIO`` plugin that makes Lightning's own checkpoint writes
    atomic. Preemption during a write is the case this exists for."""

    def save_checkpoint(self, checkpoint, path, storage_options=None) -> None:
        if storage_options is not None:
            raise TypeError(
                f"{type(self).__name__} does not accept storage_options"
            )
        atomic_torch_save(checkpoint, path)


# ------------------------------------------------------------------- RNG state


class RngStateCallback(pl.Callback):
    """Persist and restore RNG state, which Lightning 2.2.1 does not checkpoint.

    Without this, a resumed run reseeds nothing and the per-epoch DataLoader
    draws a different shuffle than the uninterrupted run would have, so
    predictions diverge even though every weight and optimizer moment was
    restored correctly.

    Lightning calls callback ``load_state_dict`` while restoring modules and
    callbacks, before the fit loop runs and before any DataLoader for the
    resumed epoch is constructed, which is the point at which the state has to
    be back in place.
    """

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "python" in state_dict:
            random.setstate(state_dict["python"])
        if "numpy" in state_dict:
            np.random.set_state(state_dict["numpy"])
        if "torch" in state_dict:
            torch.random.set_rng_state(
                state_dict["torch"].cpu()
                if isinstance(state_dict["torch"], torch.Tensor)
                else state_dict["torch"]
            )
        cuda = state_dict.get("torch_cuda")
        if cuda is not None and torch.cuda.is_available():
            devices = torch.cuda.device_count()
            if len(cuda) == devices:
                torch.cuda.set_rng_state_all([s.cpu() for s in cuda])


# ------------------------------------------------------- batch-size schedule


def epoch_shuffle_seed(train_seed: int, epoch: int) -> int:
    """Deterministic per-epoch shuffle seed, a function of nothing else."""
    payload = f"shuffle\x00{int(train_seed)}\x00{int(epoch)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31)


class StatefulStackedFvTDataModule(StackedFvTDataModule):
    """``StackedFvTDataModule`` with a resumable batch-size schedule and a
    shuffle that does not depend on ambient RNG.

    Two changes, both required for resume to reproduce an uninterrupted run.

    **Batch-size schedule.** The parent doubles ``batch_size`` whenever the
    current epoch appears in ``batch_size_milestones``. On resume the
    accumulated size has to come back, and a milestone must not be applied
    twice if the resume lands on one. Recording which milestones were already
    applied makes the schedule idempotent rather than dependent on where the
    resume point falls.

    **Shuffle.** The parent passes ``shuffle=True`` with no ``generator``, so
    ``RandomSampler`` draws its seed from the global RNG at iteration time.
    That makes the epoch's batch order a function of everything that consumed
    RNG before it -- and on resume Lightning builds the train DataLoader one
    extra time in ``setup_data()`` before ``reload_dataloaders_every_n_epochs``
    rebuilds it, so the stream ends up exactly one draw out of phase and every
    subsequent epoch shuffles differently. Restoring RNG state cannot fix that,
    because the extra draw happens after the restore.

    Seeding an explicit generator from ``(train_seed, epoch)`` removes the
    dependence entirely: an epoch's order is the same however many times
    Lightning happens to construct the loader, and whatever else drew from the
    global stream. With ``num_workers > 0`` it also makes worker seeding
    deterministic, since ``DataLoader`` derives ``base_seed`` from the
    generator when one is supplied.

    This *does* change batch order relative to the Phase 1 baseline. Phase 3
    runs are therefore not bit-comparable with Phase 1 runs; they are
    internally reproducible and resumable, which Phase 1 runs are not.
    """

    def __init__(self, *args, shuffle_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._applied_milestones: set[int] = set()
        self.shuffle_seed = int(shuffle_seed)

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        epoch = self.trainer.current_epoch
        if (
            epoch in self.batch_size_milestones
            and epoch not in self._applied_milestones
        ):
            self.batch_size *= self.batch_size_multiplier
            self._applied_milestones.add(epoch)
            print(f"Batch size updated to: {self.batch_size}", flush=True)

        generator = torch.Generator()
        generator.manual_seed(epoch_shuffle_seed(self.shuffle_seed, epoch))

        # deliberately not calling super(): it would double the batch size
        # again and would shuffle off the global RNG
        return DataLoader(
            self.stacked_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "batch_size": int(self.batch_size),
            "applied_milestones": sorted(self._applied_milestones),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.batch_size = int(state_dict["batch_size"])
        self._applied_milestones = set(state_dict.get("applied_milestones", []))


# ------------------------------------------------------- per-estimator saver


class ResumableIndividualSaver(SaveIndividualClassifierCallback):
    """The existing per-estimator saver, with its selection state persisted.

    Two changes, both narrow:

    * ``best_scores`` (and the epoch each was set at) survive a restart, so an
      interrupted run does not silently reset every estimator's best to
      infinity and overwrite a genuinely better checkpoint with a worse one.
    * the ``*_best.pt`` / ``*_last.pt`` writes go through a temporary file and
      a rename.

    The selection *rule* is deliberately unchanged, including the fact that it
    compares against ``callback_metrics`` from a callback hook and therefore
    reads the previous epoch's loss. That off-by-one is real and documented in
    Phase 1, but fixing it changes which weights get selected, which is a
    training-behaviour change and belongs to a later phase. Phase 3 only has to
    make an interrupted run match an uninterrupted one.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_epochs: dict[str, int] = {m: -1 for m in self.monitor_metrics}

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        for i, (run_name, metric) in enumerate(
            zip(self.run_names, self.monitor_metrics)
        ):
            current = trainer.callback_metrics.get(metric, None)
            if current is None:
                continue
            if hasattr(current, "item"):
                current = current.item()
            current = float(current)

            module = self._module_at(pl_module, i)
            if current < self.best_scores[metric]:
                self.best_scores[metric] = current
                self.best_epochs[metric] = int(trainer.current_epoch)
                atomic_torch_save(
                    module.state_dict(), self.save_dir / f"{run_name}_best.pt"
                )
            atomic_torch_save(
                module.state_dict(), self.save_dir / f"{run_name}_last.pt"
            )

    def _module_at(self, pl_module, i: int):
        if self.model == "FvTClassifier":
            return pl_module.fvt_classifiers[i]
        if self.model == "AttentionClassifier":
            return pl_module.attention_classifiers[i]
        raise ValueError(f"Invalid model: {self.model}")

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_scores": dict(self.best_scores),
            "best_epochs": dict(self.best_epochs),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.best_scores.update(state_dict.get("best_scores", {}))
        self.best_epochs.update(state_dict.get("best_epochs", {}))


# --------------------------------------------------------------- run layout


def group_dirs(root: pathlib.Path, campaign_id: str, group_id: str) -> dict[str, pathlib.Path]:
    """The layout issue #2 specifies for a group."""
    base = pathlib.Path(root) / campaign_id / group_id
    dirs = {
        "base": base,
        "individual_models": base / "individual_models",
        "predictions": base / "predictions",
        "metrics": base / "metrics",
        "logs": base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


class StopAfterEpoch(pl.Callback):
    """Ask the trainer to stop once a prescribed epoch has been checkpointed.

    Used to script the interruption in the resume tests. Lightning reorders
    checkpoint callbacks to run last, so requesting the stop here still lets
    ``ModelCheckpoint`` write ``last.ckpt`` for this epoch first.
    """

    def __init__(self, epoch: int):
        self.epoch = epoch

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        if trainer.current_epoch >= self.epoch:
            trainer.should_stop = True
