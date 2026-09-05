#!/usr/bin/env bash
set -euo pipefail

CELL="${1:?cell required: FIXED1K, FIXED10K, or CLOCK}"
RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
C40_MODEL="${DISTILL_C40_MODEL:-phase-d-d100m-c40}"
H64_MODEL="${DISTILL_H64_MODEL:-phase-e-c40-h64-frozen}"
WORKERS="${BENCH_WORKERS:-16}"

shutdown_host() {
  if [[ "${BENCH_SHUTDOWN:-1}" == "1" ]]; then
    sudo shutdown -h +1
  fi
}
trap shutdown_host EXIT

COMMON=(
  "DISTILL_RUN_ID=$RUN_ID"
  "DISTILL_MODEL_NAME=$C40_MODEL"
  "DISTILL_VARIANT_NAME=phase-o-c40-integer-see"
  "DISTILL_OPPONENT_MODEL_NAME=$H64_MODEL"
  "DISTILL_CANDIDATE_ANTISYMMETRIC=1"
  "DISTILL_OPPONENT_ANTISYMMETRIC=1"
  "DISTILL_CANDIDATE_SEARCH=see"
  "DISTILL_OPPONENT_SEARCH=see"
  "DISTILL_CANDIDATE_RUNTIME=integer"
  "DISTILL_OPPONENT_RUNTIME=integer"
  "BENCH_WORKERS=$WORKERS"
  "BENCH_SHUTDOWN=0"
)

case "$CELL" in
  FIXED1K)
    env "${COMMON[@]}" \
      BENCH_FIXED_NODES=1000 \
      BENCH_ROUNDS="${BENCH_1K_ROUNDS:-20}" \
      bash infra/aws/benchmark-distilled.sh
    ;;
  FIXED10K)
    env "${COMMON[@]}" \
      BENCH_FIXED_NODES=10000 \
      BENCH_ROUNDS="${BENCH_10K_ROUNDS:-10}" \
      bash infra/aws/benchmark-distilled.sh
    ;;
  CLOCK)
    env "${COMMON[@]}" \
      DISTILL_VARIANT_NAME=phase-o-c40-integer-see-clock \
      BENCH_BASE_MS="${BENCH_BASE_MS:-5000}" \
      BENCH_ROUNDS="${BENCH_CLOCK_ROUNDS:-20}" \
      bash infra/aws/benchmark-distilled.sh
    ;;
  *)
    echo "cell must be FIXED1K, FIXED10K, or CLOCK" >&2
    exit 2
    ;;
esac
