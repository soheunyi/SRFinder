#!/bin/bash
# Run a command with a private CUDA MPS server on this job's GPU(s).
#
#   speedups/with_mps.sh <command> [args...]
#
# MPS lets the worker processes that share one GPU run kernels concurrently
# instead of time-slicing between CUDA contexts. It changes scheduling only,
# not results: measured 1.6x per GPU for five resident workers, all outputs
# bit-identical. Needs the GPU in Default compute mode (n01-n03 are) and
# node-local, per-job pipe/log directories. The server stops when the command
# exits, whatever its status.
#
# Failure mode: a fatal fault in one MPS client can take down the server and
# every other client on that GPU, so treat such a failure as recoverable for
# all workers on the GPU (resume from their last completed epoch).
set -euo pipefail
base=${MPS_BASE:-${TMPDIR:-/tmp}}/mps-${SLURM_JOB_ID:-$$}
export CUDA_MPS_PIPE_DIRECTORY=$base/pipe CUDA_MPS_LOG_DIRECTORY=$base/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
stop() { echo quit | nvidia-cuda-mps-control || true; rm -rf "$base"; }
trap stop EXIT
"$@"
