#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
BASE_RUN_ID="${DISTILL_BASE_RUN_ID:-phase-c-20260902a}"
TOTAL_SHARDS="${DISTILL_TOTAL_SHARDS:-22500}"
EXPECTED_RECORDS="${DISTILL_EXPECTED_RECORDS:-90000000}"
BUILD_WORKERS="${DISTILL_BUILD_WORKERS:-32}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
REPOSITORY="${AICHESSATHON_REPOSITORY:-/home/ec2-user/aichessathon}"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
WORK_DIRECTORY="/tmp/aichessathon-phase-d-prepare-${RUN_ID}"
VENV_DIRECTORY="$WORK_DIRECTORY/venv"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
BASE_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${BASE_RUN_ID}"
RAW_DIRECTORY="$WORK_DIRECTORY/raw"
DATASET_DIRECTORY="$WORK_DIRECTORY/dataset/phase-d-extra-90m"

if ! [[ "$TOTAL_SHARDS" =~ ^[1-9][0-9]*$ && "$EXPECTED_RECORDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "shard and record counts must be positive integers" >&2
  exit 2
fi

mkdir -p "$RAW_DIRECTORY" "$DATASET_DIRECTORY"
cd "$REPOSITORY"
if [[ ! -x "$VENV_DIRECTORY/bin/python" ]]; then
  "$UV_BIN" venv --python 3.12 "$VENV_DIRECTORY"
fi
if ! "$VENV_DIRECTORY/bin/python" -c 'import chess, numpy' >/dev/null 2>&1; then
  "$UV_BIN" pip install --python "$VENV_DIRECTORY/bin/python" \
    -r training/requirements-phase-b-aws.txt
fi

for component in phase-c-base-1m phase-c-extra-2m phase-c-extra-7m; do
  aws s3 sync "$BASE_PREFIX/dataset/$component/" \
    "$RUN_PREFIX/dataset/$component/" --only-show-errors
done

aws s3 sync "$RUN_PREFIX/raw/medium/" "$RAW_DIRECTORY/" \
  --exclude "*" --include "*.jsonl.gz" --only-show-errors
mapfile -t ALL_INPUTS < <(find "$RAW_DIRECTORY" -type f -name '*.jsonl.gz' | sort)
if (( ${#ALL_INPUTS[@]} != TOTAL_SHARDS )); then
  echo "found ${#ALL_INPUTS[@]} raw shards, expected $TOTAL_SHARDS" >&2
  exit 1
fi

if ! aws s3 ls "$RUN_PREFIX/dataset/phase-d-extra-90m/dataset.json" >/dev/null 2>&1; then
  "$VENV_DIRECTORY/bin/python" -m distill.inspect_teacher \
    "${ALL_INPUTS[@]}" --expected-records "$EXPECTED_RECORDS" --expected-candidates 8 \
    >"$WORK_DIRECTORY/inspection.json"
  "$VENV_DIRECTORY/bin/python" -m distill.build_dataset \
    "${ALL_INPUTS[@]}" --output "$DATASET_DIRECTORY" \
    --records-per-shard 10_000 --workers "$BUILD_WORKERS"
  aws s3 sync "$DATASET_DIRECTORY/" \
    "$RUN_PREFIX/dataset/phase-d-extra-90m/" --only-show-errors
  aws s3 cp "$WORK_DIRECTORY/inspection.json" \
    "$RUN_PREFIX/dataset/phase-d-extra-90m/inspection.json" --only-show-errors
fi

GIT_REVISION="$(git -c safe.directory="$REPOSITORY" rev-parse HEAD)"
jq -n --arg run_id "$RUN_ID" --arg base_run_id "$BASE_RUN_ID" \
  --arg git_revision "$GIT_REVISION" --argjson records "$EXPECTED_RECORDS" \
  '{run_id:$run_id,base_run_id:$base_run_id,status:"complete",git_revision:$git_revision,new_records:$records,total_records:($records + 10000000)}' \
  >"$WORK_DIRECTORY/status.json"
aws s3 cp "$WORK_DIRECTORY/status.json" "$RUN_PREFIX/status/phase-d-prepare.json" \
  --only-show-errors

if [[ "${DISTILL_SHUTDOWN:-1}" == "1" ]]; then
  sudo shutdown -h +1
fi
