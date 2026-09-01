#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:?DISTILL_RUN_ID is required}"
MODEL_NAME="${DISTILL_MODEL_NAME:-combined}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
ROUNDS="${BENCH_ROUNDS:-10}"
WORKERS="${BENCH_WORKERS:-3}"
BASE_MS="${BENCH_BASE_MS:-5000}"
WORK_DIRECTORY="/tmp/aichessathon-benchmark-${RUN_ID}-${MODEL_NAME}"
CANDIDATE="$WORK_DIRECTORY/candidate"
FALLBACK="$WORK_DIRECTORY/fallback"
OUTPUT="$WORK_DIRECTORY/${MODEL_NAME}-vs-fallback.json"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"

mkdir -p "$CANDIDATE/weights" "$FALLBACK"
cp agent.py nnue_runtime.py search_engine.py "$CANDIDATE/"
cp agent.py nnue_runtime.py search_engine.py "$FALLBACK/"
aws s3 cp "$RUN_PREFIX/models/$MODEL_NAME/${MODEL_NAME}.npz" \
  "$CANDIDATE/weights/nnue.npz"

BENCH_AGENT="$CANDIDATE" \
BENCH_OPPONENT="$FALLBACK" \
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
