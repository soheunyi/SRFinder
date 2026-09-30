#!/bin/bash
# Throughput matrix on one GPU. Each configuration runs P concurrent workers of
# five members each, alone on the GPU, at fixed batch 1024 (1 epoch) and fixed
# batch 32768 (8 epochs). The fixed batch is a benchmark-only override; the
# production schedule is unchanged. Needs PYTHON, DATA and OUT; LOCAL is a
# node-local directory for the local-checkpoint arm (default /tmp).
#
#   PYTHON=... DATA=... OUT=... speedups/run_timing_matrix.sh
set -euo pipefail
: "${PYTHON:?}" "${DATA:?}" "${OUT:?}"
LOCAL=${LOCAL:-/tmp}/spd-${SLURM_JOB_ID:-$$}
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
mkdir -p "$OUT" "$LOCAL"
ALL3=nosync,fast_gbn,fast_reinforce
ALL4=nosync,fast_gbn,fast_reinforce,graphs

# workers NAME P CKPT(nfs|local) PATCHES
workers() {
  local name=$1 P=$2 ckpt=$3 patches=$4
  for phase in "1024 1" "32768 8"; do
    set -- $phase; local bs=$1 ep=$2
    local pids=() failed=0
    for w in $(seq 0 $((P-1))); do
      local g=$(( (w % 5) * 5 )) out=$OUT/$name/bs$bs/w$w
      local croot=$out/ckpt; [ "$ckpt" = local ] && croot=$LOCAL/$name/bs$bs/w$w
      mkdir -p "$(dirname "$out")"
      "$PYTHON" speedups/bench.py --data "$DATA" --members $g,$((g+1)),$((g+2)),$((g+3)),$((g+4)) \
        --epochs $ep --fixed-batch $bs --out "$out" --ckpt-root "$croot" --patches "$patches" > "$out.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [ "$failed" = 0 ] || return 1
  done
}
export -f workers; export OUT LOCAL PYTHON DATA

workers P1_base 1 nfs ""
workers P5_base 5 nfs ""
speedups/with_mps.sh bash -c 'workers P5_mps 5 nfs ""'
speedups/with_mps.sh bash -c 'workers P5_mps_local 5 local ""'
workers P1_all3 1 local "$ALL3"
workers P1_all4 1 local "$ALL4"
speedups/with_mps.sh bash -c "workers P5_mps_local_all3 5 local $ALL3"
speedups/with_mps.sh bash -c "workers P5_mps_local_all4 5 local $ALL4"
speedups/with_mps.sh bash -c "workers P8_mps_local_all4 8 local $ALL4"
"$PYTHON" speedups/summarize_timing.py "$OUT"
rm -rf "$LOCAL"
