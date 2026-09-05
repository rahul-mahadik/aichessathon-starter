#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model name required}"
RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
WORKERS="${BENCH_WORKERS:-16}"

if [[ ! "$MODEL" =~ ^phase-[a-zA-Z0-9-]+$ ]]; then
  echo "model must be a safe phase model name" >&2
  exit 2
fi

shutdown_host() {
  if [[ "${BENCH_SHUTDOWN:-1}" == "1" ]]; then
    sudo shutdown -h +1
  fi
}
trap shutdown_host EXIT

run_clock_match() {
  local variant="$1" opponent_runtime="$2"
  DISTILL_RUN_ID="$RUN_ID" \
  DISTILL_MODEL_NAME="$MODEL" \
  DISTILL_VARIANT_NAME="${MODEL}-${variant}" \
  DISTILL_OPPONENT_MODEL_NAME="$MODEL" \
  DISTILL_CANDIDATE_ANTISYMMETRIC=1 \
  DISTILL_OPPONENT_ANTISYMMETRIC=1 \
  DISTILL_CANDIDATE_SEARCH=strong \
  DISTILL_CANDIDATE_RUNTIME=buffered \
  DISTILL_OPPONENT_SEARCH=strong \
  DISTILL_OPPONENT_RUNTIME="$opponent_runtime" \
  BENCH_BASE_MS="${BENCH_BASE_MS:-5000}" \
  BENCH_ROUNDS="${BENCH_CLOCK_ROUNDS:-20}" \
  BENCH_WORKERS="$WORKERS" \
  BENCH_SHUTDOWN=0 \
    bash infra/aws/benchmark-distilled.sh
}

run_clock_match buffered-vs-reference-clock reference
run_clock_match buffered-vs-incremental-clock incremental
