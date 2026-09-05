#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
EVAL_RUN_ID="${DISTILL_EVAL_RUN_ID:-eval-20260901a}"
MODELS="${DISTILL_BLEND_MODELS:-phase-d-d100m-c40 phase-d-d100m-c20 phase-e-c40-h64-frozen}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
WORK_DIRECTORY="/tmp/aichessathon-blend-${RUN_ID}"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
EVAL_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${EVAL_RUN_ID}"

mkdir -p "$WORK_DIRECTORY/external" "$WORK_DIRECTORY/models" "$WORK_DIRECTORY/results"
aws s3 sync "$EVAL_PREFIX/raw/deep/" "$WORK_DIRECTORY/external/" \
  --exclude "*" --include "*.jsonl.gz" --only-show-errors
mapfile -t INPUTS < <(find "$WORK_DIRECTORY/external" -type f -name '*.jsonl.gz' | sort)
MODEL_ARGUMENTS=()
for model in $MODELS; do
  if [[ ! "$model" =~ ^phase-[a-zA-Z0-9-]+$ ]]; then
    echo "unsafe blend model name: $model" >&2
    exit 2
  fi
  path="$WORK_DIRECTORY/models/$model.npz"
  aws s3 cp "$RUN_PREFIX/models/$model/$model.npz" "$path" --only-show-errors
  MODEL_ARGUMENTS+=(--model "$path")
done

output="$WORK_DIRECTORY/results/evaluator-blend-grid.json"
uv sync --frozen --python 3.12
uv run python -m distill.sweep_evaluator_blends \
  "${INPUTS[@]}" "${MODEL_ARGUMENTS[@]}" --include-classical \
  --step "${DISTILL_BLEND_STEP:-0.1}" --output "$output"
aws s3 cp "$output" "$RUN_PREFIX/evaluator/blends/evaluator-blend-grid.json" \
  --only-show-errors

if [[ "${DISTILL_SHUTDOWN:-1}" == "1" ]]; then
  sudo shutdown -h +1
fi
