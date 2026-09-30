# Execution speedups for resident independent training (#4)

Opt-in changes to how the PR #7 training path runs. None of them changes the
recipe (float32, schedule, LR, Adam epsilon, architecture, loss), and the reported comparisons checked bit-identical finite results.
Resume and full artifact-path acceptance are separate gates.

| Change | Where | What it removes |
|---|---|---|
| MPS | `with_mps.sh` | Worker processes on one GPU time-slicing between CUDA contexts |
| `nosync` | `patches.py` | A GPU→CPU copy (and wait) after every step; one sync per parameter in `nan_check` |
| `fast_gbn` | `patches.py` | ~226 `.item()` syncs per step from GhostBatchNorm's 0-dim buffers |
| `fast_reinforce` | `patches.py` (via `phase5/slice_rewrite.py`) | Slice-and-cat interleaving in the reinforce layers |
| `graphs` | `patches.py` | Per-kernel CPU dispatch for each member's forward and backward |

## Measured

One L40, five members per worker, members of ~561k training / ~276k validation
rows. Member-epochs per second on the whole GPU; the 100-epoch column
interpolates `c + k/batch` between the two measured batch sizes over the
`[1, 3, 6, 10, 15]` milestone schedule.

| Configuration | @1,024 | @32,768 | members per GPU-hour, 100 epochs | vs current |
|---|---:|---:|---:|---:|
| 1 worker, current | 0.045 | 1.18 | 24 | 0.49x |
| 5 workers, current | 0.150 | 1.80 | 48 | 1.00x |
| 5 workers + MPS | 0.204 | 3.13 | 77 | 1.6x |
| 5 workers + MPS + `nosync`,`fast_gbn`,`fast_reinforce` | 0.238 | 3.78 | 92 | 1.9x |
| 5 workers + MPS + all four | 0.711 | 3.98 | 125 | 2.6x |
| 8 workers + MPS + all four | 0.973 | 3.87 | 127 | 2.6x |

With all patches the GPU saturates at batch 32,768, so eight workers do not
beat five there. Node-local checkpoints made no difference against NFS.

## Step 2 (AttentionClassifier)

The full 30-epoch, 15-member single-worker comparison measured 661.26 s
with `nosync,fast_gbn` and 314.20 s with graphs added: 2.10x faster fitting.
Peak allocated memory was 1.66 GB eager and 2.25 GB graphed. States, histories,
best weights and probe predictions were identical, including a fresh-process
restart at epoch 16. The concurrent exactness check ran five workers, each
repeating the same three-member subset; it did not test five 15-member ensembles.

Separate fixed-batch tests use 15 prepared members split three per worker on
one L40. The rates below **project** the 30-epoch schedule using `c + k/batch`
interpolation from batch sizes 1,024 and 32,768; they are not measured full-
schedule rates for five production 15-member ensembles.

| Configuration | projected members per GPU-hour | vs 1 worker |
|---|---:|---:|
| 1 worker, current | 72 | 1.0x |
| 5 workers, current | 228 | 3.2x |
| 5 workers + MPS + `nosync`,`fast_gbn` | 343 | 4.8x |
| 5 workers + MPS + `nosync`,`fast_gbn`,`graphs` | 1,011 | 14.1x |

The 0.52 GB peak belongs to a three-member worker at fixed batch 32,768;
do not use it as a 15-member production-worker memory bound. Actual artifact-
path integration also passed on small attention tasks. Production group-size
throughput and aggregate memory still need measurement.

## Exactness

`compare.py` checks, per member, digests of final weights and running stats,
Adam and scheduler state, the per-epoch validation-loss/LR history (hex floats),
best epoch, best weights, and logits on a validation probe.

- 3 members, 6 epochs, real schedule through batch 4,096: every patch set is
  identical to the unpatched path.
- The timing runs above (fixed batch 1,024 and 32,768, one to eight concurrent
  workers, with and without MPS): every configuration is identical to one
  unpatched worker.

Not covered here: resuming mid-schedule with patches enabled. Run the PR #7
completed-epoch resume checks with them on before adopting, since `fast_gbn`
adds state-dict save/load hooks and the graph cache is per process.

## Using them

```bash
# members from raw mother samples (reads the shared cache only)
python speedups/extract_members.py --config CR_CONFIG.yml --out MEMBERS --count 25

# one worker with every patch
python speedups/bench.py --data MEMBERS --members 0,1,2,3,4 --out RUN \
    --patches nosync,fast_gbn,fast_reinforce,graphs

# the exactness check and the throughput matrix (one GPU each)
PYTHON=python DATA=MEMBERS OUT=EXACT speedups/run_exactness.sh
PYTHON=python DATA=MEMBERS OUT=TIMING speedups/run_timing_matrix.sh

# any launcher under MPS
speedups/with_mps.sh <command that starts the workers>
```

In production code, call `patches.install([...])` in each worker process
before the model is built.

## Integration notes

- `network_blocks.py`, `fvt_classifier.py` and `independent_training.py` are in
  `train_stage._implementation()`. Editing them in place makes in-progress runs
  refuse to resume, so adopt for new tasks or keep these as opt-in patches.
- `fast_gbn` keeps every buffer registered and writes `gbs` back in
  `_save_to_state_dict`: checkpoint contents are unchanged and existing
  `*_best.pt` files load with `strict=True`.
- `graphs` does not graph Adam: `capturable=True` computes bias corrections on
  the device in float32 instead of host doubles, which is not bit-identical.
- Graphs add ~0.5 GB of GPU memory per worker. Host RAM is ~1–2 GB per worker.
- MPS: a fatal fault in one client can stop the server and every client on that
  GPU; treat it as recoverable for all of them.


## Integration review

The first extraction version omitted step from benchmark hparams, which made
initialize_members select the Step-1/2 seed policy. It also uses a random
CR-sized subset with physical weights, not the campaign's learned-region and
weight-preparation pipeline. Historical measurements remain controlled execution
proxy results; the 100-epoch throughput values are projections, not measured
full-schedule campaign throughput. Newly extracted fixtures explicitly use a
Step-3 identity under a benchmark-only experiment name.

The integration copy adds strict member-set comparison, attention history
flushing for nosync, portable reinforce functions, and a temporary
speedups.scope.execution_patches context manager. CUDA graphs apply to both
FvT (Steps 1 and 3) and the Step-2 AttentionClassifier. The scope restores imported aliases and
class methods on exit, including exceptions. Keep production adoption opt-in
until actual artifact-path resume, memory and provenance checks pass.


## Explicit artifact API

For new task roots, run_stage, run_tasks and run_campaign accept
execution_patches=['nosync','fast_gbn','fast_reinforce','graphs'].
The campaign CLI accepts --execution-patches with the same comma-separated
names for prepare/run/resume. Defaults stay empty. Patch names, implementation
hashes and the graph NaN-check timing are recorded in ownership manifests;
resume rejects a different patch configuration. Existing active runs should
continue using their original source snapshot.

The MPS wrapper uses a fresh private temporary directory per invocation and
requires one explicitly visible GPU. It propagates command failure and stops
only its private server. Shell benchmark drivers propagate failed worker exit
codes, and new benchmark runs reject an existing output directory. A completed
or partial fingerprint must never stand in for a failed rerun.
