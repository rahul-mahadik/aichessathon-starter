#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model name required}"
RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
WORKERS="${BENCH_WORKERS:-16}"
CONTROL_SEARCH="${DISTILL_CONTROL_SEARCH:-ordered}"

if [[ ! "$MODEL" =~ ^phase-[a-zA-Z0-9-]+$ ]]; then
  echo "model must be a safe phase model name" >&2
  exit 2
fi
if [[ "$CONTROL_SEARCH" != "ordered" && "$CONTROL_SEARCH" != "strong" ]]; then
  echo "DISTILL_CONTROL_SEARCH must be ordered or strong" >&2
  exit 2
fi

shutdown_host() {
  if [[ "${BENCH_SHUTDOWN:-1}" == "1" ]]; then
    sudo shutdown -h +1
  fi
}
trap shutdown_host EXIT

run_match() {
  local nodes="$1" rounds="$2"
  DISTILL_RUN_ID="$RUN_ID" \
  DISTILL_MODEL_NAME="$MODEL" \
  DISTILL_VARIANT_NAME="${MODEL}-see-vs-${CONTROL_SEARCH}" \
  DISTILL_OPPONENT_MODEL_NAME="$MODEL" \
  DISTILL_CANDIDATE_ANTISYMMETRIC=1 \
  DISTILL_OPPONENT_ANTISYMMETRIC=1 \
  DISTILL_CANDIDATE_SEARCH=see \
  DISTILL_CANDIDATE_RUNTIME=reference \
  DISTILL_OPPONENT_SEARCH="$CONTROL_SEARCH" \
  DISTILL_OPPONENT_RUNTIME=reference \
  BENCH_FIXED_NODES="$nodes" \
  BENCH_ROUNDS="$rounds" \
  BENCH_WORKERS="$WORKERS" \
  BENCH_SHUTDOWN=0 \
    bash infra/aws/benchmark-distilled.sh
}

run_match 1000 "${BENCH_1K_ROUNDS:-20}"
run_match 10000 "${BENCH_10K_ROUNDS:-5}"

DISTILL_RUN_ID="$RUN_ID" \
DISTILL_MODEL_NAME="$MODEL" \
DISTILL_VARIANT_NAME="${MODEL}-see-vs-${CONTROL_SEARCH}-clock" \
DISTILL_OPPONENT_MODEL_NAME="$MODEL" \
DISTILL_CANDIDATE_ANTISYMMETRIC=1 \
DISTILL_OPPONENT_ANTISYMMETRIC=1 \
DISTILL_CANDIDATE_SEARCH=see \
DISTILL_CANDIDATE_RUNTIME=reference \
DISTILL_OPPONENT_SEARCH="$CONTROL_SEARCH" \
DISTILL_OPPONENT_RUNTIME=reference \
BENCH_BASE_MS="${BENCH_BASE_MS:-5000}" \
BENCH_ROUNDS="${BENCH_CLOCK_ROUNDS:-20}" \
BENCH_WORKERS="$WORKERS" \
BENCH_SHUTDOWN=0 \
  bash infra/aws/benchmark-distilled.sh
