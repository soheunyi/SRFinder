# Campaign preparation and recovery

Status: instructions under validation, not a launch decision. Issue #6 controls
scientific scope and the eventual user launch. New Step-3 plans use
`sr_quantile_cr_complement_v2`: CR is every event below the SR cut, with no
lower cutoff. Earlier engineering plans retain their earlier region identities. Keep the immutable deployed source
snapshot, frozen plan, source pools and MotherSamples records available throughout
training and recovery. Never update the deployed Python files in place.

## Prepare without training

Run from the deployed snapshot's `soheun/` directory with the validated Python
environment. Set absolute paths to a new store and execution root; neither may
point at the legacy cache. Use the source-bound `frozen-plan.json`, not the inventory.

```bash
export PYTHON_BIN=/path/to/validated/environment/bin/python
export CAMPAIGN_PLAN=/absolute/path/to/frozen-plan.json
export CAMPAIGN_STORE=/absolute/path/to/new/artifact_store
export CAMPAIGN_OUTPUT=/absolute/path/to/new/execution
export NPROC=5
export EXPORT_BATCH_SIZE=32768
export EXECUTION_PATCHES=nosync,fast_gbn,fast_reinforce,graphs

"$PYTHON_BIN" phase5/campaign.py prepare --dry-run \
  --plan "$CAMPAIGN_PLAN" --store "$CAMPAIGN_STORE" --output "$CAMPAIGN_OUTPUT" \
  --device cuda --export-batch-size "$EXPORT_BATCH_SIZE" --execution-patches "$EXECUTION_PATCHES"

"$PYTHON_BIN" phase5/campaign.py prepare \
  --plan "$CAMPAIGN_PLAN" --store "$CAMPAIGN_STORE" --output "$CAMPAIGN_OUTPUT" \
  --device cuda --export-batch-size "$EXPORT_BATCH_SIZE" --execution-patches "$EXECUTION_PATCHES"
```

Review counts, source/draft/code fingerprints and pending gates before the second
command. It imports source descriptors and pins ownership, but creates no models
and submits no job. Batch 32,768 is an explicit export choice; it does not change
training batches. It was exact on the completed 37-model L40 native chain.

## Staged training after the user's launch decision

Use `all` for campaign sweeps/arrays. Standalone engineering tests may use
`--partition=statds --account=statds --qos=statds_priority`; do not assume those
settings are interchangeable with campaign-account QoS. Verify the applicable
`all` account/QoS before submission. These commands intentionally do not guess it.

Request one GPU, sufficient time and eight CPU cores. The current replacement
benchmark requests 24 GB host RAM for five workers. Treat 1–2 GB per worker as a
target, not a measured upper bound: an earlier native child peak was about 5 GB
while preparing data. Reduce the reservation only after full concurrent peaks
are measured. Worker GPU budgets share the initial free memory after a 3 GiB
safety margin; each reserves 5 GiB compute headroom. These are configurable
admission estimates, not device memory reservations. `resident=auto` stages data
on CPU if its estimate cannot fit; it does not change observations or model seeds.

Export the variables above to the allocation. After explicit user authorization,
set `CAMPAIGN_LAUNCH_AUTHORIZED=yes`, then invoke one command per staged job:

```bash
bash phase5/campaign_allocated.sh 1
bash phase5/campaign_allocated.sh 2
bash phase5/campaign_allocated.sh 3
```

Each invocation opens a private MPS server and closes it on exit. Stage 1 trains
only base ensembles; stage 2 also permits smeared ensembles; stage 3 permits CR
ensembles. All use the same prepared full scope and stable case directories.
Already verified results are reused. Do not start two coordinators on the same
execution root. Stage boundaries do not imply statistical acceptance.

## Interruption and failed-task retry

A nonzero Slurm exit is an interrupted execution, not a rejection or an accepted
model result. Inspect its log, `progress.json`, task metrics and memory decisions.
The coordinator stops other workers when a worker fails. A fatal MPS client fault
may also stop the private server and all clients; restart the allocation/server
and recover every incomplete case. Fatal MPS fault injection remains untested.

Use the same snapshot, plan, store, output, export profile and patch policy.
Rerun the interrupted stage command. Recovery restores the last **completed
 epoch** checkpoint, optimizer, schedulers, selection history and RNG state;
work since that boundary is replayed. If no valid checkpoint exists, the case
starts from its declared initialization. Mid-epoch cursor recovery is unsupported.
A corrupt or mismatched checkpoint must be investigated, not silently discarded.

For memory pressure, reduce concurrency while preserving all identities:

```bash
export NPROC=3
bash phase5/campaign_allocated.sh 3
```

Inspect or fully verify completed cases with:

```bash
"$PYTHON_BIN" phase5/campaign.py status --output "$CAMPAIGN_OUTPUT"
"$PYTHON_BIN" phase5/campaign.py status --output "$CAMPAIGN_OUTPUT" --verify
```

Only after permanent models, scores and receipts verify, preview cleanup for an
explicit case, then repeat with `--apply`. Keep incomplete checkpoints:

```bash
"$PYTHON_BIN" phase5/campaign.py cleanup --output "$CAMPAIGN_OUTPUT" --case CASE_ID
```

## Evaluation boundary still to finish

The frozen specification uses upper-only clipping, the pinned continuous-affine
kernels, B=1000 and alpha=0.05, with the signed-4b variant when required. All
member scores remain available. The fixed comparison member is seed 0. The development set is the pilot null
cells at eta 2/infinity and SR size 0.2, seeds 0–99. All three aggregation rules
are reported, with advisory 5% shape/10% extension thresholds; the user makes
the final aggregation/member-count decision before inspecting power results.

A store-backed evaluation launcher and new-output-root table/figure adapters are
still required before this runbook is complete. Do not execute the legacy master
figure runner against the old cache as a substitute: its task paths target
historical outputs. No evaluation launch command is claimed runnable yet.
