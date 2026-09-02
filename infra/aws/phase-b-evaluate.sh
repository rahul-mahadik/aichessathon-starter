#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:-scale-20260901a}"
EVAL_RUN_ID="${DISTILL_EVAL_RUN_ID:-eval-20260901a}"
MODELS="${DISTILL_MODELS:-phase-b-m phase-b-r-deep phase-b-h-medium phase-b-h-deep}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
REPOSITORY="${AICHESSATHON_REPOSITORY:-/home/ec2-user/aichessathon}"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
WORK_DIRECTORY="/tmp/aichessathon-phase-b-evaluate-${RUN_ID}"
VENV_DIRECTORY="$WORK_DIRECTORY/venv"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
EVAL_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${EVAL_RUN_ID}"

mkdir -p "$WORK_DIRECTORY/external" "$WORK_DIRECTORY/buckets" "$WORK_DIRECTORY/models"
cd "$REPOSITORY"
if [[ ! -x "$VENV_DIRECTORY/bin/python" ]]; then
  "$UV_BIN" venv --python 3.12 "$VENV_DIRECTORY"
fi
if ! "$VENV_DIRECTORY/bin/python" -c 'import chess, numba, numpy' >/dev/null 2>&1; then
  "$UV_BIN" pip install --python "$VENV_DIRECTORY/bin/python" \
    chess==1.11.2 numba==0.67.0 numpy==2.5.2
fi

aws s3 sync "$EVAL_PREFIX/raw/deep/" "$WORK_DIRECTORY/external/" \
  --exclude "*" --include "*.jsonl.gz" --only-show-errors
aws s3 sync "$RUN_PREFIX/mined/phase-b-ablation/buckets/" "$WORK_DIRECTORY/buckets/" \
  --exclude "*" --include "*-deep.jsonl.gz" --only-show-errors
mapfile -t EXTERNAL_INPUTS < <(find "$WORK_DIRECTORY/external" -type f -name '*.jsonl.gz' | sort)

for model in $MODELS; do
  if [[ ! "$model" =~ ^[a-zA-Z0-9-]+$ ]]; then
    echo "DISTILL_MODELS contains an unsafe model name: $model" >&2
    exit 2
  fi
  model_path="$WORK_DIRECTORY/models/$model.npz"
  output_directory="$WORK_DIRECTORY/results/$model"
  mkdir -p "$output_directory"
  aws s3 cp "$RUN_PREFIX/models/$model/$model.npz" "$model_path" --only-show-errors
  "$VENV_DIRECTORY/bin/python" -m distill.compare_evaluators \
    "${EXTERNAL_INPUTS[@]}" --model "$model_path" --antisymmetric \
    --output "$output_directory/external.json"
  for trace in "$WORK_DIRECTORY"/buckets/*-deep.jsonl.gz; do
    bucket="$(basename "$trace" -deep.jsonl.gz)"
    "$VENV_DIRECTORY/bin/python" -m distill.compare_evaluators \
      "$trace" --model "$model_path" --antisymmetric \
      --output "$output_directory/bucket-$bucket.json"
  done
  aws s3 sync "$output_directory/" "$RUN_PREFIX/evaluator/phase-b/$model/" \
    --only-show-errors
done

STATUS_PATH="$WORK_DIRECTORY/status.json"
jq -n --arg run_id "$RUN_ID" --arg models "$MODELS" \
  '{run_id:$run_id,models:$models,status:"complete"}' >"$STATUS_PATH"
aws s3 cp "$STATUS_PATH" "$RUN_PREFIX/status/phase-b-evaluator.json" --only-show-errors

if [[ "${DISTILL_SHUTDOWN:-1}" == "1" ]]; then
  sudo shutdown -h +1
fi
