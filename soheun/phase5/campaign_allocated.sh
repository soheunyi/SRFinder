#!/bin/bash
# Explicit manual launch inside an allocation; never calls sbatch or auto-requeues.
set -euo pipefail
stage=${1:?Pass the final stage to run: 1, 2, or 3}
case "$stage" in 1|2|3) ;; *) echo 'Stage must be 1, 2, or 3' >&2; exit 2;; esac
: "${PYTHON_BIN:?Set the validated environment interpreter}"
: "${CAMPAIGN_PLAN:?Set the frozen-plan.json path}"
: "${CAMPAIGN_STORE:?Set the new artifact store path}"
: "${CAMPAIGN_OUTPUT:?Set the prepared execution directory}"
if [[ -n "${CAMPAIGN_PILOT_TIER:-}" ]]; then
  case "$CAMPAIGN_PILOT_TIER" in A|B|all) ;; *) echo 'Pilot tier must be A, B, or all' >&2; exit 2;; esac
  if [[ "${PILOT_LAUNCH_AUTHORIZED:-no}" != yes ]]; then
    echo 'Pilot launch requires the user decision and PILOT_LAUNCH_AUTHORIZED=yes' >&2
    exit 2
  fi
  scope=(--pilot-tier "$CAMPAIGN_PILOT_TIER")
else
  if [[ "${CAMPAIGN_LAUNCH_AUTHORIZED:-no}" != yes ]]; then
    echo 'Full-campaign launch requires the user decision and CAMPAIGN_LAUNCH_AUTHORIZED=yes' >&2
    exit 2
  fi
  scope=(--start-campaign)
fi
: "${SLURM_JOB_ID:?Run inside the requested Slurm allocation}"
: "${CUDA_VISIBLE_DEVICES:?Allocate exactly one visible GPU}"
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec bash speedups/with_mps.sh "$PYTHON_BIN" phase5/campaign.py resume \
  --plan "$CAMPAIGN_PLAN" --store "$CAMPAIGN_STORE" --output "$CAMPAIGN_OUTPUT" \
  --device cuda "${scope[@]}" --through-stage "$stage" \
  --nproc "${NPROC:-5}" --resident "${RESIDENT:-auto}" \
  --safety-gib "${SAFETY_GIB:-3}" --compute-headroom-gib "${COMPUTE_HEADROOM_GIB:-5}" \
  --export-batch-size "${EXPORT_BATCH_SIZE:-32768}" \
  --execution-patches "${EXECUTION_PATCHES:-nosync,fast_gbn,fast_reinforce,graphs}"
