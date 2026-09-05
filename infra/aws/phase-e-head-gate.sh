#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?candidate model name required}"
BASE_MODEL="${DISTILL_BASE_MODEL:-phase-e-c40-h64-frozen}"
RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
WAIT_ATTEMPTS="${DISTILL_WAIT_ATTEMPTS:-480}"
WORKERS="${BENCH_WORKERS:-16}"

if [[ ! "$MODEL" =~ ^phase-[a-zA-Z0-9-]+$ ]]; then
  echo "candidate model name is unsafe" >&2
  exit 2
fi

shutdown_host() {
  sudo shutdown -h +1
}
trap shutdown_host EXIT

for ((attempt = 1; attempt <= WAIT_ATTEMPTS; attempt++)); do
  if aws s3 ls "$RUN_PREFIX/status/training-${MODEL}.json" >/dev/null 2>&1; then
    break
  fi
  if (( attempt == WAIT_ATTEMPTS )); then
    echo "$MODEL did not become ready" >&2
    exit 1
  fi
  echo "waiting for $MODEL ($attempt/$WAIT_ATTEMPTS)"
  sleep 30
done

DISTILL_MODELS="$MODEL" \
DISTILL_EVAL_NAMESPACE="phase-e-${MODEL}" \
DISTILL_SHUTDOWN=0 \
  bash infra/aws/phase-d-evaluate.sh

run_match() {
  local nodes="$1" rounds="$2"
  DISTILL_RUN_ID="$RUN_ID" \
  DISTILL_MODEL_NAME="$MODEL" \
  DISTILL_VARIANT_NAME="${MODEL}-vs-h64" \
  DISTILL_OPPONENT_MODEL_NAME="$BASE_MODEL" \
  DISTILL_CANDIDATE_ANTISYMMETRIC=1 \
  DISTILL_OPPONENT_ANTISYMMETRIC=1 \
  DISTILL_CANDIDATE_SEARCH=strong \
  DISTILL_OPPONENT_SEARCH=strong \
  DISTILL_CANDIDATE_RUNTIME=reference \
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
DISTILL_VARIANT_NAME="${MODEL}-vs-h64-clock" \
DISTILL_OPPONENT_MODEL_NAME="$BASE_MODEL" \
DISTILL_CANDIDATE_ANTISYMMETRIC=1 \
DISTILL_OPPONENT_ANTISYMMETRIC=1 \
DISTILL_CANDIDATE_SEARCH=strong \
DISTILL_OPPONENT_SEARCH=strong \
DISTILL_CANDIDATE_RUNTIME=reference \
DISTILL_OPPONENT_RUNTIME=reference \
BENCH_BASE_MS="${BENCH_BASE_MS:-5000}" \
BENCH_ROUNDS="${BENCH_CLOCK_ROUNDS:-10}" \
BENCH_WORKERS="$WORKERS" \
BENCH_SHUTDOWN=0 \
  bash infra/aws/benchmark-distilled.sh
