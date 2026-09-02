#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:-phase-c-20260902a}"
EVAL_RUN_ID="${DISTILL_EVAL_RUN_ID:-eval-20260901a}"
MODELS="${DISTILL_MODELS:-phase-c-d1m-c4p5 phase-c-d3m-c4p5 phase-c-d10m-c4p5 phase-c-d1m-c10 phase-c-d1m-c20 phase-c-d1m-c40}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
REPOSITORY="${AICHESSATHON_REPOSITORY:-/home/ec2-user/aichessathon}"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
WORK_DIRECTORY="/tmp/aichessathon-phase-c-evaluate-${RUN_ID}"
VENV_DIRECTORY="$WORK_DIRECTORY/venv"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
EVAL_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${EVAL_RUN_ID}"

mkdir -p "$WORK_DIRECTORY/external" "$WORK_DIRECTORY/models" "$WORK_DIRECTORY/results"
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
mapfile -t EXTERNAL_INPUTS < <(find "$WORK_DIRECTORY/external" -type f -name '*.jsonl.gz' | sort)
if (( ${#EXTERNAL_INPUTS[@]} == 0 )); then
  echo "external evaluator holdout is empty" >&2
  exit 1
fi

for model in $MODELS; do
  if [[ ! "$model" =~ ^phase-c-[a-zA-Z0-9-]+$ ]]; then
    echo "DISTILL_MODELS contains an unsafe model name: $model" >&2
    exit 2
  fi
  model_path="$WORK_DIRECTORY/models/$model.npz"
  output="$WORK_DIRECTORY/results/$model.json"
  aws s3 cp "$RUN_PREFIX/models/$model/$model.npz" "$model_path" --only-show-errors
  "$VENV_DIRECTORY/bin/python" -m distill.compare_evaluators \
    "${EXTERNAL_INPUTS[@]}" --model "$model_path" --antisymmetric --output "$output"
done
aws s3 sync "$WORK_DIRECTORY/results/" "$RUN_PREFIX/evaluator/phase-c/" --only-show-errors

jq -n --arg run_id "$RUN_ID" --arg models "$MODELS" \
  '{run_id:$run_id,models:$models,status:"complete"}' >"$WORK_DIRECTORY/status.json"
aws s3 cp "$WORK_DIRECTORY/status.json" "$RUN_PREFIX/status/phase-c-evaluator.json" \
  --only-show-errors

if [[ "${DISTILL_SHUTDOWN:-1}" == "1" ]]; then
  sudo shutdown -h +1
fi
