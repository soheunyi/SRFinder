# Campaign engineering validation

The campaign remains unlaunched. Scientific calibration and the final CR
aggregation/member-count decision are pending pilot diagnostics and user review.
The staged commands are in `CAMPAIGN_RUNBOOK.md`.

## Full-schedule measurements on one L40

| Workload | Measured elapsed time | Scope |
|---|---:|---|
| Step 1, 15 independent members, 100 epochs | 21.77 min | One worker; preparation, fitting, checkpoints and X1/X2 export |
| Step 2, 15 independent members, 30 epochs | 5.71 min | One worker; 106.6 s preparation, 217.2 s fitting, 17.4 s export |
| Step 3, five workers × five members, 100 epochs | 14.56 min | Includes fresh-process restart after epoch 16, preparation and full X2 export |
| Step 2, five workers × 15 members, 30 epochs | 4.90 min | Prepared real tensors, setup and reference probes; excludes encoding and full score export |

The native Step-1/2 comparison is job 210759. The Step-3 comparison is job
210728: serial 33.22 min, five workers 14.56 min, **2.281×** speedup and
**103.0 members/GPU-hour**. Best/full-X2 score receipts, final parameters and
normalization, Adam, schedulers and histories agree exactly. The five CR
selections share one mother sample. Sampled parallel device use peaked at
26.42 GiB, host RSS at 11.26 GB; resumed-run mean GPU busy time was 95.17%.

Job 210801 extends the Step-2 resource check to the production group size:
five concurrent processes, each with 15 members. All five match the reference
fingerprints; peak allocation is 2.25 GB per worker. The measured prepared-fit
rate is 918.4 members/GPU-hour. These are independent process copies of one
prepared mother-sample fixture. This rate is not an end-to-end campaign rate;
pilot measurements must include source loading, encoding and exports.

Five workers per GPU remain the default. The fixed-batch projections in
`speedups/README.md` describe earlier benchmarks and must not replace these
native measurements when estimating campaign duration.

## Correctness safeguards

- New Step-3 tasks explicitly pin the region recipe; completed legacy v1
  plans remain readable. Complement CR includes every event below the SR cut.
- Both Adam constructors explicitly use ε=1e-8. New production plans pin torch
  2.3.1.post300 and refuse a runtime mismatch before execution.
- The fixed-single-member comparison uses the original float32 log ratios,
  with no probability conversion.
- Primary evaluation and summaries require a recorded, hashed user decision.
  The frozen undecided specification remains an unmodified planning blueprint.
- Ensemble guidance requires all mother seeds 0–99 at both η=2 and infinity.
  Tier A produces descriptive diagnostics only.
- Cleanup journals preserve the original file list and completed deletions
  across interruption; permanent weights, scores and histories remain intact.
- Manuscript table rows are pivoted by η/SR and ε. Infinity power and background
  extrapolation diagnostics are explicitly internal outputs.

Small synthetic regression tests validate implementation and recovery. They do
not establish scientific calibration. Fatal MPS fault injection and end-to-end
five-worker Step-1/2 rates across distinct mother samples remain unmeasured.


The corrected snapshot passed CPU job 210811 and GPU job 210812. Coverage
includes three-stage artifact equivalence on both devices, attention graph
replay and epoch recovery, frozen-kernel evaluation, exact raw-member scoring,
interrupted cleanup, staged pilot-to-full reuse, figure consumers, the internal
background diagnostic CLI/resume, and full inventory/source-plan checks.
The regenerated scope stays at 146,537 networks and 19,105 training nodes;
internal diagnostics reuse existing training cases.


The optional five-to-fifteen member workflow is implemented separately from
the initial five-member plan. Jobs 210846 (CPU) and 210847 (GPU/MPS) checked
five original CR members plus ten added members against a fresh fifteen-member
fit on a synthetic source, including a real epoch-boundary interruption of the
added group. Receipts match exactly; original models, scores, histories and
execution files stay unchanged. Job 210848 checked CLI preparation/resume
without model creation and full-model cleanup accounting. These are two-epoch
correctness checks, not a measured production extension throughput or an
ensemble-size decision. The optional workflow requires a recorded user count
decision and a separate frozen execution using the shared immutable store.
