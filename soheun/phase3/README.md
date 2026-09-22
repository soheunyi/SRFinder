# Phase 3 — resumable one-GPU stacked training

Issue: soheunyi/SRFinder#2. Depends on Phases 1 and 2, which passed.

## What Lightning already gives you

A probe (`ckpt_probe.py`) of the installed pytorch_lightning 2.2.1 settles what
has to be built and what does not:

| requirement from the issue | status |
|---|---|
| all model parameters | free |
| all optimizer and scheduler states | free |
| current epoch | free (plus `global_step` and full loop state) |
| per-estimator best validation scores | free, *if* the callback defines `state_dict`/`load_state_dict` |
| batch-size schedule state | free, *if* the datamodule defines them (stored under the datamodule's class name) |
| RNG states | **not saved** — the one real gap |

And the reason resumption is impossible today is mundane:
`StackedFvTClassifier.fit` registers **no `ModelCheckpoint` at all**, so no
`last.ckpt` has ever been written. The `*_best.pt` files it does write hold
weights only.

## What this adds

| file | role |
|---|---|
| `resumable.py` | `RngStateCallback`, `StatefulStackedFvTDataModule`, `ResumableIndividualSaver`, `AtomicCheckpointIO`, `StopAfterEpoch` |
| `patches.py` | substitutes them into the `stacked_fvt` namespace and forwards `ckpt_path` into `trainer.fit` |
| `run_resumable.py` | runs one group into the layout the issue specifies |
| `compare_resume.py` | the five resume tests |
| `submit_resume_test.sbatch` | `--requeue`, `--signal=B:USR1@120`, one L40 |
| `ckpt_probe.py` | the probe above; rerun it when the lightning version moves |

Layout:

```
training_runs/<campaign_id>/<group_id>/
  manifest.json      identities, seeds, env, slurm job, segment history
  last.ckpt          full stacked checkpoint, written atomically
  completion.json    written only after predictions are exported
  individual_models/ per-estimator best weights, keyed by identity
  predictions/       probe logits by identity
  metrics/           fingerprint.part<N>.json, one per segment
  logs/
```

## Two findings from building it

**Per-estimator files were keyed by a timestamp.** `train_stacked_fvt` names
them `[tinfo.hash for tinfo in tinfos]`, and `utils.create_hash` derives those
from a timestamp, so the resumed process invents new names, writes its
`*_best.pt` under them and orphans the originals. Worse: with the saver's
`best_scores` restored, an estimator that never improves after the resume point
gets **no best checkpoint at all** under its current name. Phase 3 keys the
files by Phase 2's content-derived identity fingerprint instead.

**Restoring RNG state is not sufficient for a reproducible resume.** On resume
Lightning builds the train DataLoader one extra time, in `setup_data()` at the
restored epoch, before `reload_dataloaders_every_n_epochs=1` rebuilds it for the
next epoch. That throwaway build draws once from the global RNG, so every later
shuffle is one step out of phase:

```
[resumed] load_state_dict()  after = a8b88d53     restore is correct
[resumed] train_dataloader() ep=1  rng=a8b88d53   extra build at the restored epoch
[resumed] train_dataloader() ep=2  rng=b5696335   real one, stream already advanced
[whole]   train_dataloader() ep=2  rng=a8b88d53   what it should have been
```

No amount of care about *when* RNG is saved fixes this, because the extra draw
happens after the restore. `StatefulStackedFvTDataModule` therefore gives the
DataLoader an explicit generator seeded from `(train_seed, epoch)`. An epoch's
order is then the same however many times Lightning constructs the loader and
whatever else drew from the global stream, and with `num_workers > 0` worker
seeding becomes deterministic too, since `DataLoader` derives `base_seed` from a
supplied generator.

**This changes batch order relative to the Phase 1 baseline.** Phase 3 runs are
internally reproducible and resumable; they are not bit-comparable with Phase 1
and Phase 2 runs. The scientific content is unaffected — it is a different but
equally valid shuffle — but the comparison has to be Phase 3 against Phase 3.

## What is deliberately unchanged

The saver's selection *rule*, including the off-by-one Phase 1 documented: it
compares against `callback_metrics` from a callback hook and therefore reads the
previous epoch's loss, so `*_best.pt` holds epoch-N weights judged by
epoch-(N-1). Phase 3 only has to make an interrupted run match an uninterrupted
one. Changing which weights get selected is a training-behaviour change and
belongs to a later phase.

## Resume tests

- [x] Interrupt after a prescribed epoch
- [x] Resume from `last.ckpt`
- [x] Match uninterrupted optimizer and scheduler states
- [x] Match final predictions within deterministic tolerance
- [x] Preserve best checkpoints across interruption

With the shipped config (`patience: 10`) no LR reduction fires inside 20
epochs, so that gate passes without ever exercising a reduction. The scheduler
check still has teeth — `ReduceLROnPlateau` accumulates `best`,
`num_bad_epochs` and `cooldown_counter` every epoch, and a scheduler that
failed to restore would show `num_bad_epochs` reset to 0 — but `factor=0.5`
reaching `param_groups`, the reduced LR surviving in `optimizer_states`, and
the cooldown path were all untested.

`--lr-patience` overrides the patience so reductions fire and straddle the
resume point, and `--require-lr-change` makes the comparator **fail** unless
one actually did. The sbatch passes it whenever `LR_PATIENCE` is set. With
`patience=2`, interrupt after epoch 7:

```
PASS  3d. an LR reduction actually fired     distinct LRs: [0.00125, 0.0025, 0.005, 0.01]
PASS  3e. LR trajectory matches every epoch
PASS  3c. validation losses match every epoch   max |diff| = 0.000e+00 over 20 epochs
PASS  4.  final predictions match               max |diff| = 0.000e+00 over 5 estimators
PASS  5b. best checkpoints identical            all match
```

The override is test-only and is recorded in `manifest.json` as
`lr_patience_override`, so a run carries evidence of what it tested. Both sides
of a comparison must use the same value.

## Running

```bash
# harness smoke test, ~7 min
CAMPAIGN=phase3_smoke NUM_STACKS=2 MAX_EPOCHS=5 STOP_AFTER=2 \
    sbatch phase3/submit_resume_test.sbatch

# full gate
CAMPAIGN=phase3_full NUM_STACKS=5 MAX_EPOCHS=20 STOP_AFTER=7 \
    sbatch phase3/submit_resume_test.sbatch

# scheduler gate: force LR reductions across the resume point
CAMPAIGN=phase3_lrsched NUM_STACKS=5 MAX_EPOCHS=20 STOP_AFTER=7 LR_PATIENCE=2 \
    sbatch phase3/submit_resume_test.sbatch
```

The sbatch carries `--requeue` and `--signal=B:USR1@120`. Lightning installs the
SIGUSR1 handler itself — "SLURM auto-requeueing enabled" appears in every log —
and now that a `ModelCheckpoint` is registered it has something to write when
the signal arrives.
