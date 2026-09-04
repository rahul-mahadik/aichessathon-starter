#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Phase D model name required}"
RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
WORKERS="${BENCH_WORKERS:-16}"

if [[ ! "$MODEL" =~ ^phase-d-[a-zA-Z0-9-]+$ ]]; then
  echo "model must be a Phase D model name" >&2
  exit 2
fi

shutdown_host() {
  if [[ "${BENCH_SHUTDOWN:-1}" == "1" ]]; then
    sudo shutdown -h +1
  fi
}
trap shutdown_host EXIT

run_match() {
  local variant="$1" opponent_model="$2" candidate_search="$3" candidate_runtime="$4"
  local opponent_search="$5" opponent_runtime="$6" nodes="$7" rounds="$8"
  DISTILL_RUN_ID="$RUN_ID" \
  DISTILL_MODEL_NAME="$MODEL" \
  DISTILL_VARIANT_NAME="$variant" \
  DISTILL_OPPONENT_MODEL_NAME="$opponent_model" \
  DISTILL_CANDIDATE_ANTISYMMETRIC=1 \
  DISTILL_OPPONENT_ANTISYMMETRIC=1 \
  DISTILL_CANDIDATE_SEARCH="$candidate_search" \
  DISTILL_CANDIDATE_RUNTIME="$candidate_runtime" \
  DISTILL_OPPONENT_SEARCH="$opponent_search" \
  DISTILL_OPPONENT_RUNTIME="$opponent_runtime" \
  BENCH_FIXED_NODES="$nodes" \
  BENCH_ROUNDS="$rounds" \
  BENCH_WORKERS="$WORKERS" \
  BENCH_SHUTDOWN=0 \
    bash infra/aws/benchmark-distilled.sh
}

# Search-quality axis: same frozen evaluator, equal raw node budgets.
for nodes in 1000 10000; do
  rounds="${BENCH_1K_ROUNDS:-20}"
  [[ "$nodes" == "10000" ]] && rounds="${BENCH_10K_ROUNDS:-5}"
  run_match \
    "${MODEL}-strong-vs-baseline" "$MODEL" \
    strong reference baseline reference "$nodes" "$rounds"
done
if (( ${BENCH_100K_ROUNDS:-1} > 0 )); then
  run_match \
    "${MODEL}-strong-vs-baseline" "$MODEL" \
    strong reference baseline reference 100000 "${BENCH_100K_ROUNDS:-1}"
fi

# Evaluator axis inside the stronger search, held to equal node budgets.
for nodes in 1000 10000; do
  rounds="${BENCH_1K_ROUNDS:-20}"
  [[ "$nodes" == "10000" ]] && rounds="${BENCH_10K_ROUNDS:-5}"
  run_match \
    "${MODEL}-strong-incremental-vs-fallback" fallback \
    strong incremental strong reference "$nodes" "$rounds"
done
if (( ${BENCH_100K_ROUNDS:-1} > 0 )); then
  run_match \
    "${MODEL}-strong-incremental-vs-fallback" fallback \
    strong incremental strong reference 100000 "${BENCH_100K_ROUNDS:-1}"
fi

# Runtime axis: same model and search; only accumulator implementation differs.
DISTILL_RUN_ID="$RUN_ID" \
DISTILL_MODEL_NAME="$MODEL" \
DISTILL_VARIANT_NAME="${MODEL}-incremental-vs-reference-clock" \
DISTILL_OPPONENT_MODEL_NAME="$MODEL" \
DISTILL_CANDIDATE_ANTISYMMETRIC=1 \
DISTILL_OPPONENT_ANTISYMMETRIC=1 \
DISTILL_CANDIDATE_SEARCH=strong \
DISTILL_CANDIDATE_RUNTIME=incremental \
DISTILL_OPPONENT_SEARCH=strong \
DISTILL_OPPONENT_RUNTIME=reference \
BENCH_BASE_MS="${BENCH_BASE_MS:-5000}" \
BENCH_ROUNDS="${BENCH_CLOCK_ROUNDS:-20}" \
BENCH_WORKERS="$WORKERS" \
BENCH_SHUTDOWN=0 \
  bash infra/aws/benchmark-distilled.sh
