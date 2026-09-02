#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model name required}"
RUN_ID="${DISTILL_RUN_ID:-scale-20260901a}"
WORKERS="${BENCH_WORKERS:-16}"

if [[ ! "$MODEL" =~ ^phase-b-[a-z-]+$ ]]; then
  echo "model must be a phase-b model name" >&2
  exit 2
fi

run_cell() {
  local nodes="$1" rounds="$2"
  DISTILL_RUN_ID="$RUN_ID" \
  DISTILL_MODEL_NAME="$MODEL" \
  DISTILL_VARIANT_NAME="$MODEL" \
  DISTILL_CANDIDATE_ANTISYMMETRIC=1 \
  BENCH_FIXED_NODES="$nodes" \
  BENCH_ROUNDS="$rounds" \
  BENCH_WORKERS="$WORKERS" \
  BENCH_SHUTDOWN=0 \
    bash infra/aws/benchmark-distilled.sh
}

run_cell 1000 20
run_cell 10000 5
run_cell 100000 1

if [[ "${BENCH_SHUTDOWN:-1}" == "1" ]]; then
  sudo shutdown -h +1
fi
