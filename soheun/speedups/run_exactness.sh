#!/bin/bash
# Exactness check: every patch set against the unpatched path, same members,
# real milestone schedule. Variants run concurrently on one GPU (numerics do
# not depend on sharing). Needs PYTHON, DATA (member files) and OUT.
#
#   PYTHON=... DATA=... OUT=... speedups/run_exactness.sh [members] [epochs]
set -euo pipefail
: "${PYTHON:?}" "${DATA:?}" "${OUT:?}"
members=${1:-0,1,2}; epochs=${2:-6}
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
mkdir -p "$OUT"
declare -A V=([baseline]="" [nosync]="nosync" [fast_gbn]="fast_gbn" [fast_reinforce]="fast_reinforce"
              [all3]="nosync,fast_gbn,fast_reinforce" [graphs]="fast_gbn,graphs"
              [all4]="nosync,fast_gbn,fast_reinforce,graphs")
for name in "${!V[@]}"; do
  "$PYTHON" speedups/bench.py --data "$DATA" --members "$members" --epochs "$epochs" \
    --out "$OUT/$name" --patches "${V[$name]}" > "$OUT/$name.log" 2>&1 &
done
wait
"$PYTHON" speedups/compare.py "$OUT"/{baseline,nosync,fast_gbn,fast_reinforce,all3,graphs,all4}
