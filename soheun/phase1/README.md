# Phase 1 — reproduce the current scientific baseline

Issue: soheunyi/SRFinder#2, "Stabilize and benchmark resumable single-GPU
stacked CR training".

Phase 1 does **not** change training. It pins down what the current stacked CR
training actually does, so that every later phase can be checked against a
reference instead of against memory.

## What is measured

A run of the existing code path is fingerprinted through a Lightning callback
that only reads state:

| criterion (from the issue)                | fingerprint field                                      |
|-------------------------------------------|--------------------------------------------------------|
| identical initial parameters              | `init_param_digests` (sha256 per estimator state_dict)  |
| identical minibatch order                 | `epochs[].batch_order_digest`, `epochs[].batch_sizes`   |
| identical validation losses               | `val_epochs[].val_loss_per_stack`                       |
| identical selected best checkpoints       | `best.epochs`, `saver_best_scores`, `best_ckpt_digests` |
| predictions equal within tolerance        | `probe_preds_final.npy`, `probe_preds_best.npy`         |

Two extra things are recorded because they matter for later phases:

* `val_epochs[].val_loss_per_stack_as_seen_by_saver` — the `callback_metrics`
  snapshot visible while callbacks run. Lightning runs callbacks *before* the
  `LightningModule` hook that logs `val_loss_stack_i`, so this is the previous
  epoch's value. It is what `SaveIndividualClassifierCallback` compares against
  and what `ReduceLROnPlateau` steps on. Phase 1 records the staleness; it does
  not fix it.
* `env` — torch/cuDNN versions, GPU model, TF32 and matmul-precision flags,
  which bound how much bitwise agreement is reasonable to expect.

## What is *not* modified

`run_baseline.py` calls `get_step_3_tinfo_events` and `train_stacked_fvt`
directly. Two runtime wrappers are installed and removed again:

1. the fingerprint callback is appended to the list passed to
   `StackedFvTClassifier.fit`;
2. the process `chdir`s into a per-run scratch workdir for the duration of
   `fit`, so the relative paths `./data/checkpoints` and `./tb_logs` that `fit`
   hardcodes resolve inside the run directory rather than the production cache.

No production file is edited, no `TrainingInfo` record is written, and the
1.6 TiB cache is only read.

## Dev stack

The default stack is the first `--num-stacks` mother-sample seeds of the
eta = 0.1, s_SR = 0.20, null (`signal_ratio = 0`) group, i.e.

    configs/tmp/CR_fvt_training_ensemble_max_{seed}_0.0_0_0.1_0.2_0.8.yml

which is one of the two configurations the Phase 7 pilot will use. Only
`max_epochs` is overridden (default 20). That truncates the schedule without
changing any epoch's behaviour: the batch-size milestones `[1, 3, 6, 10, 15]`
and the `ReduceLROnPlateau` state are indexed by epoch, not by `max_epochs`.

## Running

```bash
# harness smoke test, minutes
NUM_STACKS=2 MAX_EPOCHS=3 OUT_ROOT=phase1_runs/smoke sbatch phase1/submit_baseline.sbatch

# reference run
sbatch phase1/submit_baseline.sbatch

# compare any two runs by hand
python phase1/compare.py phase1_runs/dev/A phase1_runs/dev/B
```

`compare.py` exits nonzero if any criterion fails. Estimators are matched by
stack position, never by `TrainingInfo` hash: `utils.create_hash` derives the
hash from a timestamp plus randomness, so the same estimator gets a different
identity on every run. Making identity content-derived is Phase 2's job.

## Exit condition

Phase 1 passes when two runs of the same dev stack agree on all five criteria.
If a criterion fails, the failure itself is the Phase 1 result: it names a
nondeterminism source that Phase 2 and Phase 3 have to remove before any
resume test or training change can be trusted.
