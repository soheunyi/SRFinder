# Phase 2 — deterministic estimator identities

Issue: soheunyi/SRFinder#2. Depends on Phase 1 having passed, which it has.

## The defect, stated precisely

`train_stacked_fvt` calls `pl.seed_everything(model_seed)` once and then
`StackedFvTClassifier.__init__` constructs `num_stacks` classifiers in a loop.
Estimator *i* therefore gets whatever the global RNG holds after *i*
constructions.

For the eta=0.1 pilot group this is not a subtlety, it is the whole story.
Inspecting the 100 configs of that group:

| identity component | source key | values across the group |
|---|---|---|
| mother-sample seed | `dataset.seed` | 0..99 |
| model-initialization seed | `CR_fvt.model_seed` | **0 for all 100** |
| training-order seed | `CR_fvt.train_seed` | **0 for all 100** |
| data-split seed | `CR_fvt.data_seed` | **0 for all 100** |
| ensemble-member seed | upstream, see below | n/a at step 3 |
| smearing-noise seed | `smearing.seed` (step 2) | **0 for all members** |

So the only thing distinguishing the 100 CR estimators is which mother-sample
subsample they see. Nothing in the config distinguishes their initializations,
and position in the stack supplies the difference by accident. Regrouping a
campaign, changing `N_RUNFILES`, or retrying a subset silently re-initializes
every estimator in it.

Upstream, the 15 SR-stats ensemble members carry
`model_seed == train_seed == data_seed == k` for k in 0..14 — three names for
one number. Phase 2 keeps them as three fields so separating them later is a
config change rather than a code change.

## What this adds

| file | role |
|---|---|
| `identity.py` | `EstimatorIdentity` (six named seeds + upstream fingerprint), `derive_seed`, `identity_from_step3_config`, `group_fingerprint` |
| `init_from_identity.py` | `apply_identities` — re-initializes a constructed stack from identity, in place, under an isolated RNG |
| `test_identity.py` | the three checkboxes, on CPU, no data, seconds |
| `run_with_identities.py` | end-to-end: trains a stack with identity-derived initialization |
| `compare_permutation.py` | end-to-end gate, keyed by identity rather than position |
| `submit_permutation.sbatch` | one L40; unit gate, then the same five estimators trained in both stack orders |

`derive_seed` is BLAKE2b over a canonical JSON encoding of the identity,
domain-separated by purpose (`model_init`, `training_order`, `data_split`,
`smearing_noise`). It does not use Python's `hash`, so it is stable across
processes, interpreter versions and machines — unlike `utils.create_hash`,
which derives `TrainingInfo` identity from a timestamp plus randomness and
hands the same estimator a different name on every run.

`isolated_rng` saves and restores both CPU and CUDA RNG state around each
construction. Without that, re-initializing a stack would leave the global RNG
somewhere different and the training-order sampler would draw a different
shuffle — a supposedly position-independent change leaking into batch order.

## Checkboxes

- [x] Every estimator has an explicit initialization seed
- [x] Stack permutation does not change estimator initialization
- [x] Repeating a group reproduces its initial state exactly

`test_identity.py` also asserts the *negative* control: that the unpatched
production path **is** position dependent. If that assertion ever starts
failing, the initialization path changed underneath and Phase 2's premise needs
re-reading.

The issue asks that reordering change neither initial parameters *nor
predictions*. Initial parameters are covered by the unit gate; predictions need
a real training run, which is what `submit_permutation.sbatch` does.

## Running

```bash
# unit gate, CPU, seconds
srun --partition=all --cpus-per-task=2 --mem=16G \
    python phase2/test_identity.py

# end-to-end permutation gate, one L40
sbatch phase2/submit_permutation.sbatch

# harness smoke test
NUM_STACKS=2 MAX_EPOCHS=3 OUT_ROOT=phase2_runs/smoke sbatch phase2/submit_permutation.sbatch
```

## Not yet wired into production

`apply_identities` is applied by the Phase 2 runner through the same
`fit`-wrapper trick Phase 1 uses. `train_stacked_fvt` is untouched and its
signature is unchanged. Switching the campaign over to identity-derived
initialization changes every estimator's weights relative to the existing
cache, so it is a deliberate decision that belongs with the Phase 8 gate, not
a side effect of landing this layer.
