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
workers on failure without writing false completion. Full manuscript-source and
result-registry binding remains integration work.

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

The full serial/parallel comparison, representative-scale CR measurements, GPU
tests of the new stage APIs, measured headroom, real-data end-to-end export, full campaign
binding and scientific acceptance gates remain open in #4 and #6. Passing the
CPU suite or merging this patch does not establish full-campaign readiness.


## Binding tasks and numerical settings

`bound_tasks` registers trusted local mother-selection records with fingerprints
and reconstructs worker contexts from JSON recipes. These shared source records
remain required inputs; no per-member index arrays or legacy cache writes are
introduced. `materialize_task` checks a declared node's dataset, member seeds,
count, epoch schedule and optional architecture/depth against supplied recipes,
then verifies completed parents and selects their models by seed. Wrong-eta or
missing-member upstreams are rejected. It defines X1 regions but never trains or
submits work. The caller must supply frozen dataset/member recipes and populate
the full campaign's source/result registry.

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
about four hours; its full serial comparison remains pending. K100 and current
stage-API GPU checks remain open. No campaign or statistical pilot is launched.
