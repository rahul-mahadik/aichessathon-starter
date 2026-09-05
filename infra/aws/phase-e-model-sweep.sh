#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?candidate model name required}"
OPPONENT="${2:-phase-d-d100m-c20}"
RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
WORKERS="${BENCH_WORKERS:-16}"

for name in "$MODEL" "$OPPONENT"; do
  if [[ "$name" != "fallback" && ! "$name" =~ ^phase-[a-zA-Z0-9-]+$ ]]; then
    echo "model must be fallback or a safe phase model name: $name" >&2
    exit 2
  fi
done

shutdown_host() {
  if [[ "${BENCH_SHUTDOWN:-1}" == "1" ]]; then
    sudo shutdown -h +1
  fi
}
trap shutdown_host EXIT

run_match() {
  local label="$1"
  local nodes="$2"
  local rounds="$3"

  DISTILL_RUN_ID="$RUN_ID" \
  DISTILL_MODEL_NAME="$MODEL" \
  DISTILL_VARIANT_NAME="${MODEL}-${label}" \
  DISTILL_OPPONENT_MODEL_NAME="$OPPONENT" \
  DISTILL_CANDIDATE_ANTISYMMETRIC=1 \
  DISTILL_OPPONENT_ANTISYMMETRIC=1 \
  DISTILL_CANDIDATE_SEARCH=strong \
  DISTILL_CANDIDATE_RUNTIME=reference \
  DISTILL_OPPONENT_SEARCH=strong \
  DISTILL_OPPONENT_RUNTIME=reference \
  BENCH_BASE_MS="${BENCH_BASE_MS:-5000}" \
  BENCH_FIXED_NODES="$nodes" \
  BENCH_ROUNDS="$rounds" \
  BENCH_WORKERS="$WORKERS" \
  BENCH_SHUTDOWN=0 \
    bash infra/aws/benchmark-distilled.sh
}

run_match fixed-1k 1000 "${BENCH_1K_ROUNDS:-20}"
run_match fixed-10k 10000 "${BENCH_10K_ROUNDS:-10}"
run_match fixed-100k 100000 "${BENCH_100K_ROUNDS:-2}"

DISTILL_RUN_ID="$RUN_ID" \
DISTILL_MODEL_NAME="$MODEL" \
DISTILL_VARIANT_NAME="${MODEL}-clock" \
DISTILL_OPPONENT_MODEL_NAME="$OPPONENT" \
DISTILL_CANDIDATE_ANTISYMMETRIC=1 \
DISTILL_OPPONENT_ANTISYMMETRIC=1 \
DISTILL_CANDIDATE_SEARCH=strong \
DISTILL_CANDIDATE_RUNTIME=reference \
DISTILL_OPPONENT_SEARCH=strong \
DISTILL_OPPONENT_RUNTIME=reference \
BENCH_BASE_MS="${BENCH_BASE_MS:-5000}" \
BENCH_ROUNDS="${BENCH_CLOCK_ROUNDS:-20}" \
BENCH_WORKERS="$WORKERS" \
BENCH_SHUTDOWN=0 \
  bash infra/aws/benchmark-distilled.sh
