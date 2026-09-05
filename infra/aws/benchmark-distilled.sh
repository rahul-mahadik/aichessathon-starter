#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:?DISTILL_RUN_ID is required}"
MODEL_NAME="${DISTILL_MODEL_NAME:-combined}"
VARIANT_NAME="${DISTILL_VARIANT_NAME:-$MODEL_NAME}"
OPPONENT_MODEL_NAME="${DISTILL_OPPONENT_MODEL_NAME:-fallback}"
CANDIDATE_ANTISYMMETRIC="${DISTILL_CANDIDATE_ANTISYMMETRIC:-0}"
OPPONENT_ANTISYMMETRIC="${DISTILL_OPPONENT_ANTISYMMETRIC:-0}"
CANDIDATE_SEARCH="${DISTILL_CANDIDATE_SEARCH:-baseline}"
CANDIDATE_RUNTIME="${DISTILL_CANDIDATE_RUNTIME:-reference}"
OPPONENT_SEARCH="${DISTILL_OPPONENT_SEARCH:-baseline}"
OPPONENT_RUNTIME="${DISTILL_OPPONENT_RUNTIME:-reference}"
CANDIDATE_PONDER="${DISTILL_CANDIDATE_PONDER:-0}"
OPPONENT_PONDER="${DISTILL_OPPONENT_PONDER:-0}"
CANDIDATE_VALUE_SCALE_CP="${DISTILL_CANDIDATE_VALUE_SCALE_CP:-1000}"
OPPONENT_VALUE_SCALE_CP="${DISTILL_OPPONENT_VALUE_SCALE_CP:-1000}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
ROUNDS="${BENCH_ROUNDS:-10}"
WORKERS="${BENCH_WORKERS:-3}"
BASE_MS="${BENCH_BASE_MS:-5000}"
FIXED_NODES="${BENCH_FIXED_NODES:-}"
WORK_DIRECTORY="/tmp/aichessathon-benchmark-${RUN_ID}-${VARIANT_NAME}-vs-${OPPONENT_MODEL_NAME}"
CANDIDATE="$WORK_DIRECTORY/candidate"
OPPONENT="$WORK_DIRECTORY/opponent"
if [[ -n "$FIXED_NODES" ]]; then
  OUTPUT="$WORK_DIRECTORY/${VARIANT_NAME}-vs-${OPPONENT_MODEL_NAME}-nodes-${FIXED_NODES}.json"
else
  OUTPUT="$WORK_DIRECTORY/${VARIANT_NAME}-vs-${OPPONENT_MODEL_NAME}.json"
fi
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"

for mode in "$CANDIDATE_SEARCH" "$OPPONENT_SEARCH"; do
  if [[ "$mode" != "baseline" && "$mode" != "strong" && "$mode" != "ordered" && "$mode" != "policy" && "$mode" != "see" && "$mode" != "frontier" ]]; then
    echo "search mode must be baseline, strong, ordered, policy, see, or frontier: $mode" >&2
    exit 2
  fi
done
for mode in "$CANDIDATE_RUNTIME" "$OPPONENT_RUNTIME"; do
  if [[ "$mode" != "reference" && "$mode" != "integer" && "$mode" != "incremental" && "$mode" != "buffered" ]]; then
    echo "runtime mode must be reference, integer, incremental, or buffered: $mode" >&2
    exit 2
  fi
done

mkdir -p "$CANDIDATE/weights" "$OPPONENT/weights"
cp agent.py nnue_runtime.py search_engine.py strong_search_engine.py ordered_search_engine.py \
  policy_search_engine.py see_search_engine.py frontier_search_engine.py "$CANDIDATE/"
cp agent.py nnue_runtime.py search_engine.py strong_search_engine.py ordered_search_engine.py \
  policy_search_engine.py see_search_engine.py frontier_search_engine.py "$OPPONENT/"
aws s3 cp "$RUN_PREFIX/models/$MODEL_NAME/${MODEL_NAME}.npz" \
  "$CANDIDATE/weights/nnue.npz"
candidate_antisymmetric=false
if [[ "$CANDIDATE_ANTISYMMETRIC" == "1" ]]; then
  candidate_antisymmetric=true
fi
candidate_ponder=false
if [[ "$CANDIDATE_PONDER" == "1" ]]; then
  candidate_ponder=true
fi
jq -n \
  --argjson antisymmetric "$candidate_antisymmetric" \
  --argjson ponder "$candidate_ponder" \
  --argjson value_scale_cp "$CANDIDATE_VALUE_SCALE_CP" \
  --arg search "$CANDIDATE_SEARCH" \
  --arg runtime "$CANDIDATE_RUNTIME" \
  '{antisymmetric:$antisymmetric,search:$search,runtime:$runtime,ponder:$ponder,value_scale_cp:$value_scale_cp}' \
  >"$CANDIDATE/weights/evaluator.json"
export BENCH_CANDIDATE_MODEL="$VARIANT_NAME"
export BENCH_CANDIDATE_SEARCH="$CANDIDATE_SEARCH"
export BENCH_CANDIDATE_RUNTIME="$CANDIDATE_RUNTIME"
export BENCH_CANDIDATE_VALUE_SCALE_CP="$CANDIDATE_VALUE_SCALE_CP"
export BENCH_CANDIDATE_SHA256
BENCH_CANDIDATE_SHA256="$(sha256sum "$CANDIDATE/weights/nnue.npz" | cut -d ' ' -f 1)"
opponent_antisymmetric=false
if [[ "$OPPONENT_ANTISYMMETRIC" == "1" ]]; then
  opponent_antisymmetric=true
fi
opponent_ponder=false
if [[ "$OPPONENT_PONDER" == "1" ]]; then
  opponent_ponder=true
fi
jq -n \
  --argjson antisymmetric "$opponent_antisymmetric" \
  --argjson ponder "$opponent_ponder" \
  --argjson value_scale_cp "$OPPONENT_VALUE_SCALE_CP" \
  --arg search "$OPPONENT_SEARCH" \
  --arg runtime "$OPPONENT_RUNTIME" \
  '{antisymmetric:$antisymmetric,search:$search,runtime:$runtime,ponder:$ponder,value_scale_cp:$value_scale_cp}' \
  >"$OPPONENT/weights/evaluator.json"
if [[ "$OPPONENT_MODEL_NAME" != "fallback" ]]; then
  aws s3 cp "$RUN_PREFIX/models/$OPPONENT_MODEL_NAME/${OPPONENT_MODEL_NAME}.npz" \
    "$OPPONENT/weights/nnue.npz"
  export BENCH_OPPONENT_SHA256
  BENCH_OPPONENT_SHA256="$(sha256sum "$OPPONENT/weights/nnue.npz" | cut -d ' ' -f 1)"
fi
export BENCH_OPPONENT_MODEL="$OPPONENT_MODEL_NAME"
export BENCH_OPPONENT_SEARCH="$OPPONENT_SEARCH"
export BENCH_OPPONENT_RUNTIME="$OPPONENT_RUNTIME"
export BENCH_OPPONENT_VALUE_SCALE_CP="$OPPONENT_VALUE_SCALE_CP"

BENCH_AGENT="$CANDIDATE" \
BENCH_OPPONENT="$OPPONENT" \
BENCH_ROUNDS="$ROUNDS" \
BENCH_BASE_MS="$BASE_MS" \
BENCH_FIXED_NODES="$FIXED_NODES" \
BENCH_WORKERS="$WORKERS" \
BENCH_OUTPUT="$OUTPUT" \
AICHESSATHON_ARTIFACTS_URI= \
  bash infra/aws/benchmark.sh
aws s3 cp "$OUTPUT" "$RUN_PREFIX/benchmarks/$(basename "$OUTPUT")"

if [[ "${BENCH_SHUTDOWN:-1}" == "1" ]]; then
  sudo shutdown -h +1
fi
