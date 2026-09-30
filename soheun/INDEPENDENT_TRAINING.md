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
