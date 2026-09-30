# Training artifact storage contract — schema 1

The tested implementation is on `codex/issue4-independent-training-core` in
draft PR #7, not merged into master. This document describes the current
float32 format. The BF16 compute/cache prototype has not changed that contract.

## Layout and identities

```text
artifact_store/
  records/<artifact_id>.json
  blobs/<payload_sha256>.weights
  blobs/<payload_sha256>.npy

execution/
  frozen-plan.json
  training-plan.json
  prepared.json
  registry/
    training-plan.json
    cases/<sha256_of_canonical_case_id>.json
  tasks/<logical_case_id>/
    stage-completion.json
    execution-metrics.json
    worker-metrics.json
    training/
      training-plan.json
      training-inputs.json
      training-completion.json
      last.ckpt
      models/<context_identity>_best.pt
  progress.json
  completion.json
```

Records contain `schema`, `kind`, `identity`, `id` and `payload`.
Semantic IDs hash canonical JSON of schema/kind/identity. Payloads separately
record SHA256, filename, byte count and, for arrays, shape/dtype. Writes are
atomic and immutable: conflicting reuse fails; identical payload bytes deduplicate.
Use `read()`, `payload_path()` and `load_array()` to check identity and bytes.
Never infer IDs from legacy filenames or overwrite existing records.

## Records and retention

| Kind | Content |
| --- | --- |
| dataset | Ordered raw-pool fingerprints, mother-selection fingerprints/parameters and row count; pools are referenced, not copied. |
| split | Versioned domain/seed/ordering recipe plus reconstructed-index count/checksum. No per-member index array. |
| model | One member's best float32 state, context/stage, training/init/numerical/code policies, split IDs and best epoch/loss. |
| training_history | Model-bound validation loss, LR and batch size for each epoch, checked against best selection. |
| scores | One member's unclipped float32 `logit_4b - logit_3b`, represented as `log_density_ratio`; shape [events], with owner/split/inference profile. |
| event_metadata | Shared ordered pool/row IDs, class, signed physical weight and simulation truth for one source/domain. |
| region | Paired upstream members, X1 score references, quantile/aggregation recipe and frozen log_tau_s/log_tau_c. |
| stage_completion | Verified model/history/required-domain score IDs, event metadata and payload checksums. |
| stage_task / case_result | Frozen source/member/upstream task and its verified result for one logical case under a plan. |
| ensemble | Optional ordered member IDs plus an explicit aggregation rule; individual members remain authoritative. |

Retain every member's best weights and all X2 scores. Steps 1/2 also retain X1
scores to define regions; Step 3 saves full X2, including CR/SR/neither.
Individual member files permit independent retries and later membership changes.
The runner enforces the declared member set; extending it needs an explicit new
plan/result binding instead of overwriting an old receipt.

Derive cheap log-psi and ensemble averages in RAM. Optional derived caches are
explicit operations, not the source of truth. Current aggregation choices are
`mean_probability`, `mean_log_density_ratio` and `mean_density_ratio`;
these are different estimators and the scientific choice remains undecided.

Permanent model artifacts hold best weights. One rolling completed-epoch
`last.ckpt` stores optimizer/scheduler/RNG/callback recovery state.
Working checkpoints and best-file copies currently remain after export until
verified cleanup is implemented. Do not delete them before validating permanent
weights, required exports and receipts. No legacy cache cleanup is performed.

## Events, split reconstruction and dtype

Physical metadata fields are `pool:uint32`, `pool_row:int64`,
`is_4b:bool`, `weight:float64` (signed), and `is_signal:bool`.
The double-precision physical weights are not double-precision NN/score caches.

A pool index refers to the dataset descriptor's ordered pool list. Within a
pinned source/split, pool/row identifies event order. For cross-mother or
cross-version joins, resolve it to `(raw_pool_fingerprint, pool_row)`;
a bare pool index is not a global identity. Never take absolute physical weights.

Outer/member indices regenerate from versioned recipes, seeds and verified
sources. Shared MotherSamples selection records are still required: regenerating
the mother masks themselves solely from seeds has not been established.
Use X1 thresholds with aligned X2 scores. Apply the analysis cap to derived
log-psi when constructing test inputs, not to stored raw member predictions.

## Consumer API

From `soheun/` in the repository environment:

```python
from pathlib import Path
import json
from artifacts.training_store import TrainingStore
from artifacts.case_registry import CaseRegistry
from artifacts.campaign_reader import CampaignReader

def open_reader(execution_path):
    execution = Path(execution_path)
    plan = json.loads((execution / "frozen-plan.json").read_text())
    manifest = json.loads((execution / "training-plan.json").read_text())
    store = TrainingStore(manifest["store"])
    registry = CaseRegistry(store, plan, execution / "registry", resume=True)
    return store, CampaignReader(registry)

# Caller chooses an explicit declared case_id.
# store, reader = open_reader("my_execution")
# pairs = reader.member_score_ids(case_id, "X2")
# scores = [store.load_array(score_id, "scores")
#           for model_id, score_id in pairs]
# gamma, reduction = reader.aggregate(
#     case_id, "X2", aggregation="mean_probability")
# arrays, audit = reader.affine_inputs(
#     case_id, aggregation="mean_probability")
```

Select subsets with `member_seeds=[...]`; group position is not member identity.
Mean probability above is an example, not an adopted final rule. Returned
metadata identifies the exact member/score IDs, reduction, case and region.

Low-level readers can access `stage_completion.identity`, which includes
ordered model_ids/history_ids, evaluation_splits, score_ids by domain and
event_metadata_ids. Validate receipts with `verify_stage_completion()` and a
verified source context. `TRAINING_COMPLETE_EXPORT_PENDING` is insufficient:
downstream use requires `STAGE_ARTIFACTS_COMPLETE` and registry validation of
source, frozen member recipe and registered parents.

CPU/GPU synthetic checks cover corruption, pairing, coverage, explicit reduction
and completed-epoch recovery. A bounded native-schedule three-stage chain also
passed training/export/registry/reader checks, including raw/representation CR.

A receipt proves artifact completion, not statistical calibration or full-campaign
readiness. Five simultaneous attention/CR ensembles, all draft figure consumers,
final cleanup policy and scientific acceptance remain open. BF16 adoption needs
an explicit format/version decision and prediction/region/reweighting checks.
Prototype uint16 bit files must never be interpreted as ordinary float16 arrays.
