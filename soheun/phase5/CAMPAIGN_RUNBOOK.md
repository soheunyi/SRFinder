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
settings are interchangeable with campaign-account QoS. Verified on 2026-09-30: use `--partition=all --account=statds --qos=normal`.
The association's default QoS can be `statds_priority`, while `all` allows only
`normal`, so specify `--qos=normal` explicitly. Recheck if cluster policy changes.

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

After the user chooses, record `status: USER_DECISION_RECORDED`, `primary_rule`
and a `decision_reference` in a
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

## Pilot and full-draft figures

After evaluation summaries and null diagnostics complete, use the new figure
entry point. It reads the recorded user decision and checks the frozen consumer
hashes before producing a complete output set:

```bash
"$PYTHON_BIN" phase5/render_campaign_figures.py \
  --execution "$CAMPAIGN_OUTPUT" --scope pilot-A \
  --decision /absolute/path/to/recorded-user-decision.json \
  --summary /absolute/new/summaries/pilot-A \
  --diagnostics /absolute/new/null-development/tier-A \
  --output /absolute/new/figures/pilot-A
```

For the full draft use `--scope draft`, the full evaluation summary, null
diagnostics covering seeds 0–99 and `--assets /absolute/path/to/pinned/figures`.
The three static assets (toy, contribution diagram, workflow) are copied only
when their checksums match the frozen plan. The two inline TikZ figures remain
in the pinned manuscript. All 19 generated figure names are checked at the end.
Use `--device cuda` for frozen-encoder t-SNE feature extraction; its two 20,000-
event embeddings retain perplexity 20, 500 iterations and random state 42.
Sampling now has fixed seed 42 and canonical order for reproducibility.

Individual tools remain available: `render_draft_efficiency.py` covers the eight
100-mother efficiency figures; `render_campaign_pulls.py` enforces the draft's
64 bins and null seeds 0–49; `render_campaign_illustrations.py` covers classifier,
overlap, calibration, tails and original/representation comparisons. Classifier
members are selected by seeds 5/13 and receive neutral seed labels. Calibration
uses the caption's 50 equal-count quantile bins and physical-weighted means.
Ratio error bars use squared event weights. These display corrections are
recorded in the per-figure JSON; bootstrap and training recipes are unchanged.

The native single-ensemble benchmark measured Step 1: 1,306.26 s including export
(15 members, 100 epochs); Step 2: 342.74 s (15 members, 30 epochs). Both matched
reference states and full X1/X2 arrays. This does not establish five concurrent
15-member ensembles. The full CR comparison measured 103 members/GPU-hour with
five workers and restart, and about 11.3 GB sampled aggregate host RSS. Keep
24 GB host reservations until the pilot provides additional workload peaks.
Final aggregation/member count and full-campaign launch remain user decisions.


Internal η=infinity power remains in `HH4b_eta_inf_power_internal.csv`.
It is excluded from manuscript tables and power figures. The manuscript table
uses `supplementary_noise_scale_table_rows.tex`, pivoted by η/SR and ε.

Derive the internal background-extrapolation comparison after the matching
training cases complete; this does not require an aggregation decision:

```bash
"$PYTHON_BIN" phase5/diagnose_extrapolation_bias.py \
  --execution "$CAMPAIGN_OUTPUT" --scope pilot-all \
  --output /absolute/new/internal/extrapolation-pilot --device cuda
```

Use `--scope full` in a separate new output root after the full campaign to
include all four SR sizes. `--resume` verifies existing case records and figures.
The CSV reports count-error mean and sample SD over mother seeds; the JSON also
keeps every 64-bin pull profile for member 0 and all three aggregation rules.
Null figures compare η=2 and infinity, with one panel per available SR size.
Signal-cell count errors use background 4b only. Their pull numerator excludes
signal while the variance includes all 4b, matching legacy `pull_bg4b`.
Undefined pulls remain explicit missing values. Partial seed coverage is marked
as partial and never supplies an ensemble-selection recommendation.

Production plans pin torch `2.3.1.post300` and Adam ε=1e-8. Runtime mismatches
refuse execution before any task starts. Tier-A ensemble-selection diagnostics
are descriptive only; guidance requires all seeds 0–99 at both η values.


## Optional five-to-fifteen extension

The initial plan remains five CR members. Only after the user's count decision,
write a separate JSON record with `status: USER_DECISION_RECORDED`,
`step3_member_count: 15`, and the actual `decision_reference`. The optional
extension uses a new plan and execution directory sharing the original immutable
store. It leaves the original plan, registry, models, scores and histories intact.

From the updated source snapshot, prepare without training:

```bash
"$PYTHON_BIN" phase5/extend_campaign_members.py \
  --origin "$CAMPAIGN_OUTPUT" --members 15 \
  --decision /absolute/path/to/recorded-count-decision.json \
  --plan-out /absolute/new/extension-15-plan.json \
  --output /absolute/new/extension-15-execution \
  --scope pilot-all --device cuda --export-batch-size 32768
```

Repeat inside a one-GPU allocation, under the private MPS wrapper, adding
`--resume --execute` to train only missing members. Five worker processes remain
the default; `--nproc` may decrease on resume. The original coordinator must be
stopped: its ownership lock is held throughout extension. Resolve any incomplete
original case before extending its member count. Existing completed upstreams
are registered under the new plan without refitting. Completed CR cases preserve
seeds 0–4 and fit/export only seeds 5–14; originally unstarted CR cases train all
15. Frozen regions, source/split recipes, numerical settings and export profiles
must match. Original-feature/representation methodological single-member cases
remain unchanged.

The extended completion references all 15 best models, per-member histories and
scores. Completed extension cleanup removes only the added members' redundant
working files after the full receipt verifies. It does not remove the original
members or their artifacts. Use the extended execution for fifteen-member
analysis and keep the original execution available for five-member comparisons.
The source used for an original in-progress run must remain unchanged; a new
extension executable cannot be used to resume that original run.
