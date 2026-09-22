"""Install the Phase 3 resumable machinery into an existing run.

Everything is substituted into the ``stacked_fvt`` module namespace and removed
again afterwards, so ``StackedFvTClassifier.fit`` executes exactly as written:
its seeding, its logger, its progress bar, its truncation and its datamodule
construction all still happen on the same lines.

Three substitutions:

* ``StackedFvTDataModule`` -> ``StatefulStackedFvTDataModule``, so the
  batch-size schedule is checkpointed.
* ``SaveIndividualClassifierCallback`` -> ``ResumableIndividualSaver``, with its
  save directory forced to ``individual_models/`` and its file names taken from
  Phase 2 identities, so best-score state survives a restart and the writes are
  atomic.
* ``pl.Trainer`` -> a factory that adds ``ModelCheckpoint(save_last=True)``, the
  RNG-state callback and an atomic ``CheckpointIO``, and that forwards
  ``ckpt_path`` into ``trainer.fit``.

``fit`` registers no ``ModelCheckpoint`` today, which is the whole reason there
is nothing to resume from.
"""

from __future__ import annotations

import pathlib
from typing import Any, Callable, Sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import stacked_fvt

from resumable import (
    AtomicCheckpointIO,
    ResumableIndividualSaver,
    RngStateCallback,
    StatefulStackedFvTDataModule,
    StopAfterEpoch,
)


def install(
    ckpt_dir: pathlib.Path,
    individual_models_dir: pathlib.Path,
    run_names: Sequence[str],
    shuffle_seed: int,
    resume_from: pathlib.Path | None = None,
    stop_after_epoch: int | None = None,
) -> Callable[[], None]:
    """Patch, and return a callable that undoes the patch.

    ``run_names`` are the names the per-estimator files are keyed by. Pass
    Phase 2 identity fingerprints: ``train_stacked_fvt`` would otherwise pass
    ``[tinfo.hash for tinfo in tinfos]``, and ``utils.create_hash`` derives
    those from a timestamp, so a resumed process would write its ``*_best.pt``
    under fresh names and orphan the ones the first process wrote. With the
    saver's ``best_scores`` restored from the checkpoint, an estimator that
    never improves after the resume point would then have no best checkpoint
    under its current name at all.
    """
    ckpt_dir = pathlib.Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    individual_models_dir = pathlib.Path(individual_models_dir)
    individual_models_dir.mkdir(parents=True, exist_ok=True)
    stable_names = list(run_names)

    class _DataModule(StatefulStackedFvTDataModule):
        """Injects the per-epoch shuffle seed, which ``fit`` does not pass."""

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("shuffle_seed", shuffle_seed)
            super().__init__(*args, **kwargs)

    orig_dm = stacked_fvt.StackedFvTDataModule
    orig_saver = stacked_fvt.SaveIndividualClassifierCallback
    orig_trainer = pl.Trainer

    class _Saver(ResumableIndividualSaver):
        def __init__(self, **kwargs: Any) -> None:
            monitor_metrics = kwargs["monitor_metrics"]
            if len(stable_names) != len(monitor_metrics):
                raise ValueError(
                    f"{len(stable_names)} run names for "
                    f"{len(monitor_metrics)} monitored metrics"
                )
            super().__init__(
                save_dir=individual_models_dir,
                run_names=stable_names,
                monitor_metrics=monitor_metrics,
                model=kwargs.get("model", "FvTClassifier"),
            )

    class _ResumingTrainer(orig_trainer):
        def fit(self, *args: Any, **kwargs: Any):
            kwargs.setdefault(
                "ckpt_path", str(resume_from) if resume_from is not None else None
            )
            return super().fit(*args, **kwargs)

    def _trainer_factory(**kwargs: Any):
        callbacks = list(kwargs.pop("callbacks", None) or [])
        callbacks.append(RngStateCallback())
        if stop_after_epoch is not None:
            callbacks.append(StopAfterEpoch(stop_after_epoch))
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                save_last=True,
                save_top_k=0,
                every_n_epochs=1,
                save_on_train_epoch_end=True,
            )
        )
        plugins = list(kwargs.pop("plugins", None) or [])
        plugins.append(AtomicCheckpointIO())
        return _ResumingTrainer(callbacks=callbacks, plugins=plugins, **kwargs)

    stacked_fvt.StackedFvTDataModule = _DataModule
    stacked_fvt.SaveIndividualClassifierCallback = _Saver
    pl.Trainer = _trainer_factory

    def uninstall() -> None:
        stacked_fvt.StackedFvTDataModule = orig_dm
        stacked_fvt.SaveIndividualClassifierCallback = orig_saver
        pl.Trainer = orig_trainer

    return uninstall
