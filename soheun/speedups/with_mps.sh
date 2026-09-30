#!/bin/bash
# One private MPS server for an explicitly visible GPU and command lifetime.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Run inside a one-GPU allocation}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "Expected exactly one visible GPU" >&2
  exit 2
fi
task_mps_parent="${MPS_BASE:-${TMPDIR:-/tmp}}"
mkdir -p "$task_mps_parent"
task_mps_dir=$(mktemp -d "$task_mps_parent/srfinder-mps-${SLURM_JOB_ID:-local}-XXXXXX")
export CUDA_MPS_PIPE_DIRECTORY="$task_mps_dir/pipe"
export CUDA_MPS_LOG_DIRECTORY="$task_mps_dir/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
stop() {
  result=$?
  trap - EXIT
  printf 'quit\n' | timeout 10s nvidia-cuda-mps-control >/dev/null 2>&1 || true
  rm -rf -- "$task_mps_dir"
  exit "$result"
}
trap stop EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
nvidia-cuda-mps-control -d
"$@"
