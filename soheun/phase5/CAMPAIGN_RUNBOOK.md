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

## Pilot without changing the full execution scope

Prepare the full manifest once using the commands above. After its prerequisites
pass and the user-authorized pilot is ready, restrict scheduling to tier A:

```bash
export CAMPAIGN_PILOT_TIER=A
export PILOT_LAUNCH_AUTHORIZED=yes
bash phase5/campaign_allocated.sh 1
bash phase5/campaign_allocated.sh 2
bash phase5/campaign_allocated.sh 3
```

Tier A covers 50 base, 50 smeared and 100 CR ensembles (2,000 networks). It uses
HH4b, the five declared signal ratios, eta 2/infinity, SR 0.2 and mother seeds
0–9. `CAMPAIGN_PILOT_TIER=B` selects seeds 10–99; `all` selects 0–99. Advance only
according to the pilot procedure. The same full manifest and store are reused.
A completed tier is `CAMPAIGN_PREFIX_COMPLETE` with `selected_scope_complete`
true, and each trained case has its own `STAGE_ARTIFACTS_COMPLETE` receipt.
It does not claim the full campaign is complete.

After the separate full-campaign launch decision, unset `CAMPAIGN_PILOT_TIER`
and set `CAMPAIGN_LAUNCH_AUTHORIZED=yes`. Completed pilot cases are verified and
reused; changing the scheduling subset does not change model identities.

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
Rerun the interrupted stage command. Recovery restores the last **completed epoch** checkpoint, optimizer, schedulers, selection history and RNG state;
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

## Evaluation after the user's aggregation decision

The fixed comparison member is seed 0. The development set is the pilot null
cells at eta 2/infinity and SR size 0.2, seeds 0–99. Count/shape diagnostics for
all three aggregation rules inform the user decision; the 5%/10% thresholds are
advisory. Run the pre-power diagnostics inside an allocation after the selected null
models complete:

```bash
"$PYTHON_BIN" phase5/diagnose_campaign_ensemble.py \
  --execution "$CAMPAIGN_OUTPUT" --tier A --device cuda \
  --output /absolute/new/null-development/tier-A
```

Use `--resume` after interruption and `--tier all` for the full 100-seed
comparison. X1 CR predictions needed for the existing train-quantile binning are
computed transiently from best weights; they do not expand permanent score
storage. The output records all member errors, all rule metrics, correlations
and the advisory choice. K=15 projections assume unchanged bias and an
exchangeable linear-averaging variance model; the nonlinear aggregation rules
make that approximation, not a measured guarantee.
Do not inspect power to choose the primary aggregation rule.

After the user chooses, record `primary_rule` and a `decision_reference` in a
JSON file. The reference identifies the actual recorded decision; do not invent
one. Choose from `single`, `mean_probability`, `mean_log_density_ratio`, or
`mean_density_ratio`. Evaluation runs all four as primary/secondary comparisons
unless an explicit smaller `--rules` list is requested.

Build the pinned kernel inside an appropriate CPU allocation:

```bash
g++ -O2 -std=c++17 -fPIC -shared -fno-fast-math -ffp-contract=off \
  run_files/affine_envelope_kernel.cpp -o run_files/affine_envelope_kernel.so

"$PYTHON_BIN" phase5/select_campaign_scope.py --plan "$CAMPAIGN_PLAN" \
  --scope pilot-A --output /absolute/new/pilot-A-cases.json

"$PYTHON_BIN" phase5/evaluate_campaign.py --execution "$CAMPAIGN_OUTPUT" \
  --case-file /absolute/new/pilot-A-cases.json \
  --decision /absolute/path/to/recorded-user-decision.json \
  --output /absolute/new/evaluation/pilot-A
```

Add `--resume` to reuse verified completed test records after interruption.
Changing the decision, frozen recipe, NumPy version or compiled binary refuses
reuse. The evaluation never changes model weights or stored score arrays.
It applies upper-only clipping, B=1000, alpha=0.05, seed 1729, and automatically
uses the signed-4b kernel when the retained physical weights require it.
Input hashes, event order, normalization/clipping diagnostics, kernel hashes and
maximizing nuisance intervals are retained with each result.

Independent CPU jobs can use disjoint case lists and separate evaluation roots.
Do not point concurrent evaluators at one output root. Merge all completed
shards by repeating `--evaluation`:

```bash
"$PYTHON_BIN" phase5/summarize_campaign_evaluation.py \
  --evaluation /absolute/new/evaluation/pilot-A \
  --scope pilot-A --output /absolute/new/summaries/pilot-A

"$PYTHON_BIN" phase5/render_campaign_power.py \
  --summary /absolute/new/summaries/pilot-A \
  --output /absolute/new/figures/pilot-A
```

The summary refuses duplicate, missing or extra cases/rules against the selected
frozen scope. It writes all-rule summaries and primary-rule table cells/CSVs;
infinity cells remain separate until manuscript placement is decided. Pilot
power PDFs have a pilot prefix. These tools never write into the legacy figure
or cache directories. Use `--tex` to enable the manuscript's external TeX fonts
when that environment is available.

The signal-concentration, pull and remaining illustration adapters and final
measured resource settings are still
needed before this runbook is complete. Do not substitute the legacy master
figure runner, whose paths target historical outputs.
