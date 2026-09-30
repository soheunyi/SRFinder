# Independent training and score artifacts

Each estimator keeps its own observations, seeds, normalization, optimizer and
scheduler. Grouping changes execution, not estimator ownership.

## Training behavior

- The FvT and attention helpers use independent streams. Training splits align
  separately to 32 rows; validation retains every row. A final 32-row train tail
  merges into the preceding batch so GhostBatchNorm has enough observations.
- Steps 1/2 initialize each member from its explicit model_seed. Step 3 retains
  the established identity-derived initialization mapping. The recorded policy
  distinguishes these recipes from historical position-dependent initialization.
- Best weights are selected in on_validation_end against their own validation
  loss. Legacy cached artifacts are not rewritten.
- GPU residency is opt-in via dataloader.preload_to_gpu. It requires independent
  batches, CUDA, zero loader workers and no pinned CPU loading in the effective
  loader configuration. Dataset memory must fit on the device.
- Epoch-boundary stream recovery verifies seeds, identities, lengths and batch
  schedule. Mid-epoch checkpoint saving is rejected; recover from the last
  completed-epoch checkpoint.
- Optional FvT vectorization remains experimental. Separate-model execution is
  the reference; this change does not establish numerical equivalence of vmap.

## Artifact interface

TrainingStore writes immutable semantic records and content-checked payloads to
an explicit new store root. It never changes the legacy cache. Model payloads
are kept separately from float32 member scores; split records contain versioned
reconstruction recipes, row counts and ordering fingerprints, not index arrays.

export_member_scores verifies the model payload and reconstructed evaluation
ordering, exports raw two-class logit differences, and checks complete finite
float32 coverage. Repeated requests can reuse verified scores under the same
inference settings; explicit recomputation remains available for validation.
Callers must pin any additional model-factory/feature-transformation semantics
in their model and inference recipes.

Aggregation is an explicit choice: mean probability, mean log-density ratio, or
mean density ratio. aggregate_scores computes values in memory. Only the
explicit cache_aggregate_scores operation persists a derived analysis cache.
Raw member scores remain authoritative. This is a storage/export interface;
campaign-specific event metadata and multi-stage orchestration remain integration
work, and no cleanup of existing checkpoints is performed by this PR.

## Validation summary and limits

Completed development checks cover independent row coverage and ordering,
standalone/group/reordered initialization, FvT/attention epoch-boundary resume,
current-epoch checkpoint selection, known-ratio loss semantics, artifact
integrity, and raw-score export alignment/cache reuse.

A real three-estimator 20-epoch residency comparison measured 1.529x speedup with
exact agreement of the checked states and histories. A separate short Step-1
five-process experiment measured about 2.98x at batch 1,024 and 1.33x at batch
32,768, with exact checked outputs. One completed model also exported its entire
held-out sample with exact agreement on the direct verification batch. These
are scoped development results, not full-campaign performance or calibration
claims. Raw experiment history is intentionally kept outside this review.

The full 100-epoch concurrency/restart comparison and representative Step-2
measurement are pending. Production-scale CR, end-to-end campaign integration,
and statistical acceptance gates remain open in #4 and #6. The process runner
and campaign launch tools are separate work and are not delivered by this core
patch. Neither merging this core patch nor passing its component tests authorizes
or establishes readiness for a full retraining campaign.

Relevant tests live in phase5/test_independent_streams.py,
phase5/test_training_alignment.py, phase5/test_checkpoint_selection.py,
phase5/test_member_initialization.py, phase5/test_training_store.py,
phase5/test_export_scores.py and phase5/test_weight_convention.py.

## New upstream and test-input adapters

Recipe-based loading supports FvT and attention model artifacts. Step-2 contexts
can use those encoder artifacts with an explicitly validated source context,
without looking up a legacy encoder record or writing a legacy TrainingInfo
pickle. Source dataset/version verification remains the caller's responsibility.

Stream fingerprints include the feature recipe (including smearing and encoder
mode). The expanded fingerprint is versioned; older running checkpoints should
continue with their original source snapshot, rather than silently accepting a
changed feature dataset.

The in-memory affine-input adapter preserves signed 4b weights, canonical event
ordering and the frozen X1-defined threshold. It uses a common log-domain scale
for 3b weights and checks both endpoint normalizers. Synthetic agreement with
the existing signed bootstrap reference is an interface check, not a claim of
real-data calibration.


## Source, region and member-split reconstruction

The verified source adapter checks raw-pool and mother-selection fingerprints
before reconstructing X1/X2 from the native seed recipe. The region adapter
pairs base and smeared models, freezes thresholds from X1 scores, and checks
X2 ordering before classifying SR/CR/neither. Original-feature Step-3 contexts
select only frozen CR rows. Representation-based CR diagnostics remain separate
integration work.

Member splits reproduce each stage's native shuffle/split order, storing only
recipes, counts and ordered fingerprints. Recipes pin implementation and library
versions and exclude legacy auxiliary score arrays. Step-1/2 source/split checks
also passed against a real multi-pool sample; this does not validate a complete
real-data three-stage retraining workflow.

The synthetic three-stage test trains two members for two epochs per stage,
registers best weights and member scores, defines regions on X1, trains CR
members, exports all X2 rows, and prepares test inputs without running bootstrap
inference. It checks reconstructed rows against actual training tensors, caller
RNG preservation and rejection of swapped split receipts across multiple pools.
These interfaces do not yet provide a full campaign launcher or a validated
cleanup/completion protocol.

Additional tests: phase5/test_source_context.py,
phase5/test_artifact_regions.py, phase5/test_artifact_step3_context.py and
phase5/test_three_stage_artifacts.py (requires --out pointing to a new directory).


## Resumable stage worker and export

run_stage connects verified contexts to training, required-domain export and a
checked stage-completion receipt. It does not select configurations, submit
jobs or launch a campaign. Training and export can resume independently: a
completed training group reuses its best models after an export failure. Output
locks and recipe/source fingerprints reject concurrent or mismatched reuse.

Steps 1/3 share raw tensor banks with member-specific index views; Step 2 keeps
its member-specific transformed inputs. Checkpoints retain epoch-boundary state
and history. Permanent per-member validation/LR records must agree with the
selected best epoch/loss. Completion also verifies all required scores and
physical event IDs and binds model, score and metadata payload checksums.
Working checkpoints are retained; no cleanup of existing data is performed.

The exporter keeps all X1/X2 scores for Steps 1/2 and all X2 for Step 3. It uses
the recorded Step-2 encoder and explicit feature/code, batch-size and numerical
profiles. Contiguous inference inputs remove a small float32 discrepancy found
between different tensor layouts. Verified cache hits skip feature reads and
inference. Signed physical weights and pool/raw-row IDs stay in shared metadata;
raw member scores stay float32 and unaggregated.

For CUDA, callers supply device_budget_bytes and compute_headroom_bytes.
resident=True requires data to fit; resident='auto' stages data on CPU if it
does not fit. worker_budgets divides one coordinator snapshot among five workers
by default, after an explicit safety margin. These are admission estimates, not
memory reservations. Headroom and fallback still need real GPU validation; no
model identities, observations, batch sizes or schedules change with placement.
The process-pool and full dependency-graph scheduler remain integration work.

The expanded synthetic test checks all three stages against uninterrupted
reference training, stops/restarts at an epoch boundary, interrupts export,
forbids refitting on export retry, compares direct predictions exactly, and
rejects incomplete coverage or corrupted histories. CPU checks do not establish
GPU throughput, safe production headroom or statistical calibration.
