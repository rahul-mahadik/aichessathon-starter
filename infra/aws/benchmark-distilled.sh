#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:?DISTILL_RUN_ID is required}"
MODEL_NAME="${DISTILL_MODEL_NAME:-combined}"
VARIANT_NAME="${DISTILL_VARIANT_NAME:-$MODEL_NAME}"
OPPONENT_MODEL_NAME="${DISTILL_OPPONENT_MODEL_NAME:-fallback}"
CANDIDATE_ANTISYMMETRIC="${DISTILL_CANDIDATE_ANTISYMMETRIC:-0}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
ROUNDS="${BENCH_ROUNDS:-10}"
WORKERS="${BENCH_WORKERS:-3}"
BASE_MS="${BENCH_BASE_MS:-5000}"
WORK_DIRECTORY="/tmp/aichessathon-benchmark-${RUN_ID}-${VARIANT_NAME}-vs-${OPPONENT_MODEL_NAME}"
CANDIDATE="$WORK_DIRECTORY/candidate"
OPPONENT="$WORK_DIRECTORY/opponent"
OUTPUT="$WORK_DIRECTORY/${VARIANT_NAME}-vs-${OPPONENT_MODEL_NAME}.json"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"

mkdir -p "$CANDIDATE/weights" "$OPPONENT"
cp agent.py nnue_runtime.py search_engine.py "$CANDIDATE/"
cp agent.py nnue_runtime.py search_engine.py "$OPPONENT/"
aws s3 cp "$RUN_PREFIX/models/$MODEL_NAME/${MODEL_NAME}.npz" \
  "$CANDIDATE/weights/nnue.npz"
if [[ "$CANDIDATE_ANTISYMMETRIC" == "1" ]]; then
  printf '{"antisymmetric":true}\n' >"$CANDIDATE/weights/evaluator.json"
fi
export BENCH_CANDIDATE_MODEL="$VARIANT_NAME"
export BENCH_CANDIDATE_SHA256
BENCH_CANDIDATE_SHA256="$(sha256sum "$CANDIDATE/weights/nnue.npz" | cut -d ' ' -f 1)"
if [[ "$OPPONENT_MODEL_NAME" != "fallback" ]]; then
  mkdir -p "$OPPONENT/weights"
  aws s3 cp "$RUN_PREFIX/models/$OPPONENT_MODEL_NAME/${OPPONENT_MODEL_NAME}.npz" \
    "$OPPONENT/weights/nnue.npz"
  export BENCH_OPPONENT_SHA256
  BENCH_OPPONENT_SHA256="$(sha256sum "$OPPONENT/weights/nnue.npz" | cut -d ' ' -f 1)"
fi
export BENCH_OPPONENT_MODEL="$OPPONENT_MODEL_NAME"

BENCH_AGENT="$CANDIDATE" \
BENCH_OPPONENT="$OPPONENT" \
BENCH_ROUNDS="$ROUNDS" \
BENCH_BASE_MS="$BASE_MS" \
BENCH_WORKERS="$WORKERS" \
BENCH_OUTPUT="$OUTPUT" \
AICHESSATHON_ARTIFACTS_URI= \
  bash infra/aws/benchmark.sh
aws s3 cp "$OUTPUT" "$RUN_PREFIX/benchmarks/$(basename "$OUTPUT")"

if [[ "${BENCH_SHUTDOWN:-1}" == "1" ]]; then
  sudo shutdown -h +1
fi
