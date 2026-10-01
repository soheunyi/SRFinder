# Independent training and score artifacts

Each estimator keeps its own observations, seeds, normalization, optimizer and
scheduler. Grouping changes execution, not estimator ownership. This draft
provides training/artifact APIs; it does not launch a campaign.

## Training and recovery

- FvT and attention helpers use independent streams. Training splits align
  separately to 32 rows, while validation retains all rows. A final 32-row train
  tail merges into the preceding batch for GhostBatchNorm.
- Steps 1/2 initialize from each member's model_seed. Step 3 retains the existing
  identity-derived initialization mapping. The recorded policy distinguishes
  these recipes from historical position-dependent initialization.
- Best weights use their own current validation loss. Epoch-boundary checkpoints
  preserve model, Adam, scheduler, RNG, stream and history state. Mid-epoch
  checkpoint saving is rejected.
- Source/recipe/numerical fingerprints reject incompatible recovery. Existing
  running checkpoints must retain their original source snapshot.
- Optional weight vectorization remains experimental; separate-model execution
  is the reference. This patch does not establish vmap equivalence.

## Stage APIs

`artifacts.run_stage.run_stage` connects verified Step-1/2/3 contexts to training,
required-domain export and a checked completion receipt. Output locks prevent
concurrent writers. An export failure resumes from completed best models without
refitting. Reopening a completed stage verifies its artifacts before reuse.

Steps 1/3 share raw tensor banks with member-specific index views. Step 2 keeps
each member's encoded/smeared features. `train_stage` publishes best weights only
after the configured schedule finishes; its marker explicitly leaves export
pending. Per-member validation/LR histories must agree with best-epoch selection.

The stage APIs do not choose scientific configurations, submit Slurm jobs, run
bootstrap inference or delete checkpoints. The spawn-process coordinator accepts explicit JSON task recipes with five
workers by default, supports a lower worker count on resume, and stops its own
workers on failure without writing false completion. The bounded dependency runner and case registry consume a caller-supplied frozen
plan; draft-specific inventory and deployment remain separate integration work.

## Data placement

The existing training helpers expose opt-in residency through
`dataloader.preload_to_gpu`, with zero data-loader workers and no pinned CPU
loading in the effective resident configuration.

The artifact stage APIs accept `resident=False`, `True` or `'auto'`. CUDA callers
must supply `device_budget_bytes` and `compute_headroom_bytes`. Forced residency
requires data to fit; auto placement uses CPU staging when it does not. Neither
choice changes model identities, observations, batch sizes or schedules.

`worker_budgets` divides one coordinator free-memory snapshot among five workers
by default, after an explicit safety margin. Counts include shared raw banks,
member index tensors and separate Step-2 transformed data. These are admission
estimates, not CUDA reservations or OOM guarantees. Workload-specific headroom,
concurrent budget assignment and GPU fallback behavior remain to be validated.

## Immutable artifacts and source provenance

`TrainingStore` writes semantic JSON records and content-checked payloads to an
explicit new store. It never rewrites legacy caches. Models and float32 member
log ratios are separate payloads. Split records retain versioned reconstruction
recipes, counts and ordered fingerprints, not index arrays.

The source adapter verifies raw pools and mother selections before reconstructing
X1/X2. Member reconstruction preserves each stage's native shuffle/split order,
pins code/library versions and excludes auxiliary score arrays from recipes.
Region definitions pair base/smeared models and freeze thresholds on X1; X2
classification checks model identities, score ordering and inference profiles.
Step-3 contexts select only frozen CR rows. The representation diagnostic uses
one frozen upstream encoder, splits raw CR rows before encoding, and applies no
smearing to those training features.

## Export and completion

`export_stage` retains all X1/X2 member scores for Steps 1/2 and all X2 for Step 3.
Step-2 inference uses each member's recorded encoder. Feature implementation,
contiguous tensor layout, inference batch size and numerical settings are explicit.
Verified cache hits skip raw-feature loading and inference. Shared metadata keeps
signed physical weights, class labels, pool/raw-row IDs and the configured
signal-pool flag. Raw member scores remain unclipped float32 logit differences.

Aggregation is explicit: mean probability, mean log-density ratio or mean density
ratio. `aggregate_scores` computes in memory; only `cache_aggregate_scores`
explicitly persists a derived cache. Raw member predictions remain authoritative.

Stage completion verifies model payloads, histories, complete required-domain
scores and physical event ordering, and binds their payload checksums. It is not
a campaign-readiness or calibration decision. Recovery checkpoints and working
files remain available; automatic cleanup is not implemented.

The affine-input adapter preserves signed 4b weights, canonical ordering and the
frozen threshold, uses a common log-domain scale for 3b weights, and checks both
endpoint normalizers before handing inputs to the existing statistical test.

## Validation and remaining gates

The CPU suite covers alignment, standalone/group/reordered initialization,
checkpoint selection, independent-stream recovery, known-ratio loss semantics,
model/source/region checks, artifact integrity, score export and signed-bootstrap
input preparation. The synthetic three-stage test checks interrupted training
against uninterrupted references, interrupted export, reuse with refitting
forbidden, exact direct predictions, complete coverage and corruption rejection.
Budget-boundary tests cover worker allocation and staged fallback.

Separate real-data checks verified Step-1 physical tensors and Step-2 row order on
a multi-pool sample. A three-estimator, 20-epoch residency comparison measured
1.529x speedup with exact checked states. A short five-process Step-1 test measured
about 2.98x at batch 1,024 and 1.33x at batch 32,768, with exact checked outputs.
These are scoped results; raw experimental history is excluded from this review.

The full Step-1 serial/five-process comparison passed exact checked states and
predictions across 150 NNs and 100 epochs: 8.19 versus 4.01 hours, a 2.04x speedup
across separate L40 allocations. Synthetic GPU stage and dependency/recovery
checks passed. Representative K100 CR, production headroom, real-data end-to-end
acceptance, draft plotting integration and scientific gates remain open in #4
and #6. Passing the
CPU suite or merging this patch does not establish full-campaign readiness.


## Binding tasks and numerical settings

`bound_tasks` registers trusted local mother-selection records with fingerprints
and reconstructs worker contexts from JSON recipes. These shared source records
remain required inputs; no per-member index arrays or legacy cache writes are
introduced. `materialize_task` checks a declared node's dataset, member seeds,
count, epoch schedule and optional architecture/depth against supplied recipes,
then verifies completed parents and selects their models by seed. Wrong-eta or
missing-member upstreams are rejected. It defines X1 regions but never trains or
submits work. The caller supplies a frozen plan with dataset/member templates and source
bindings. The registry checks completed tasks against that plan and their
registered parents before exposing results.

CUDA stage calls use the validated driver's medium matmul setting, permit cuDNN
TF32 and disable cuDNN benchmarking. Parameters and cached scores stay float32;
internal matmul precision can be reduced on supported hardware. Settings are
recorded and scoped, restored after normal return or failure, and CPU settings
are preserved. Stage calls serialize within a process because these flags are
global; training concurrency uses separate processes.

The CPU tests additionally cover materialized tasks, the representation CR path,
reordered parent completions, missing members, wrong eta/depth, failed-worker
recovery and exact restoration of runtime settings. The representative Step-2
15-member/30-epoch residency test passed exact checked states, histories and
prediction probes, with about 2.11x fit/checkpoint speedup excluding shared feature
preparation. The full five-process Step-1 arm completed 150 NNs for 100 epochs in
about four hours; its full serial comparison passed. Current stage-API and
dependency GPU smoke checks passed. K100 and the native real-data chain remain
open. No campaign or statistical pilot is launched.


## Frozen plans, bounded execution and analysis readers

`campaign_recipes.recipes` expands frozen native templates into explicit member
recipes. `CaseRegistry` rejects wrong sources, parents, seeds, schedules or
optimizer recipes and conflicting results. Completed-result reuse first verifies
all content/source inputs; a bounded process-local metadata cache checks referenced
file stamps and raises on changes. Raw-pool digests are reused only while their
file stamps match. No event tensors are kept in this cache.

`campaign_runtime.run_campaign` admits at most the requested worker count, waits
for verified dependencies, and keeps stable per-case output roots. The default
is five processes; resume can use fewer without changing scientific identities.
A prefix limit stops after complete case transactions. A failed worker terminates
only this coordinator's workers; completed-epoch checkpoints remain recoverable.
Execution timing and memory telemetry live separately from immutable receipts.

From `soheun/`, the CLI is `python phase5/campaign.py`. `prepare --dry-run` inspects
a supplied plan without writing files or submitting jobs. `prepare` imports source
descriptors and freezes plan/runtime/scope ownership. `status --verify` rechecks
registered outputs. Explicit run/resume requires either a small declared
`--validation --case ID` scope or `--start-campaign`. The latter is an execution
interface and does not replace scientific acceptance or the user's launch
instruction. CUDA execution requires explicit safety and per-worker compute
headroom; output/store paths and inference batch size are fixed for recovery.

`CampaignReader` selects scores by declared member seed, checks common ordering,
and requires a named aggregation. It derives held-out regions and signed-weight
affine-test inputs from the recorded X1 thresholds and X2 scores without saving
derived arrays. This supplies analysis primitives; migration of every draft
figure reader remains separate work. No aggregation choice or statistical pilot
is made by these APIs.

`phase5/test_campaign_runtime.py` runs two synthetic sources/eight logical cases
through actual recipes, materialization, spawned training/export, registry and
readers. CPU and GPU checks cover exact serial/spawn receipts, bounded dispatch,
completed-prefix and interrupted-epoch recovery, completed reuse without workers,
frozen optimizer rejection, changed artifact/source rejection, seed-based score
selection, explicit aggregation, and CLI dry-run/status behavior. These checks
are implementation acceptance, not calibration results or campaign launch.


## Optional execution optimizations

The speedups module supplies nosync, fast_gbn, fast_reinforce and CUDA graphs
through an explicit execution_patches argument. The campaign CLI accepts
--execution-patches with the same comma-separated names. Defaults are empty.
Prepare/run/resume must use the same named policy; ownership manifests record
its implementation hashes and reject incompatible recovery. Graphs accelerate
FvT forward/backward; Adam and attention-model training remain eager.

The policy runs in a temporary process-local scope so reused workers restore
their original functions afterward. The private MPS wrapper is optional and
manages only its own server. A fatal MPS client failure still requires recovery
of affected workers from completed-epoch checkpoints.

Eight-case artifact-pipeline checks passed with and without MPS: baseline and
optimized receipts, final/best model state, Adam/scheduler/history, and actual
interrupted-epoch recovery agreed exactly. These checks use a small fixture.
The reported 2.6x full-schedule throughput gain is a projection from fixed-batch
proxy measurements; native full-schedule performance remains to be measured.
