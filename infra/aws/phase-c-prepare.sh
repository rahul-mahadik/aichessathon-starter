#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:-phase-c-20260902a}"
BASE_RUN_ID="${DISTILL_BASE_RUN_ID:-scale-20260901a}"
BASE_SHARDS="${DISTILL_BASE_SHARDS:-500}"
TOTAL_SHARDS="${DISTILL_TOTAL_SHARDS:-2250}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
REPOSITORY="${AICHESSATHON_REPOSITORY:-/home/ec2-user/aichessathon}"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
WORK_DIRECTORY="/tmp/aichessathon-phase-c-prepare-${RUN_ID}"
VENV_DIRECTORY="$WORK_DIRECTORY/venv"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
BASE_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${BASE_RUN_ID}"
RAW_DIRECTORY="$WORK_DIRECTORY/raw"
DATASET_DIRECTORY="$WORK_DIRECTORY/dataset"

if ! [[ "$BASE_SHARDS" =~ ^[1-9][0-9]*$ && "$TOTAL_SHARDS" =~ ^[1-9][0-9]*$ ]] || (( BASE_SHARDS >= TOTAL_SHARDS )); then
  echo "DISTILL_BASE_SHARDS and DISTILL_TOTAL_SHARDS must define two nonempty components" >&2
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

aws s3 sync "$BASE_PREFIX/dataset/phase-b-m-base/" \
  "$RUN_PREFIX/dataset/phase-c-base-1m/" --only-show-errors
aws s3 cp "$BASE_PREFIX/models/phase-b-m/phase-b-m.npz" \
  "$RUN_PREFIX/models/phase-c-d1m-c4p5/phase-c-d1m-c4p5.npz" --only-show-errors
aws s3 cp "$BASE_PREFIX/models/phase-b-m/phase-b-m.json" \
  "$RUN_PREFIX/models/phase-c-d1m-c4p5/phase-c-d1m-c4p5.json" --only-show-errors

aws s3 sync "$RUN_PREFIX/raw/medium/" "$RAW_DIRECTORY/" \
  --exclude "*" --include "*.jsonl.gz" --only-show-errors
mapfile -t ALL_INPUTS < <(find "$RAW_DIRECTORY" -type f -name '*.jsonl.gz' | sort)
if (( ${#ALL_INPUTS[@]} != TOTAL_SHARDS )); then
  echo "found ${#ALL_INPUTS[@]} raw shards, expected $TOTAL_SHARDS" >&2
  exit 1
fi

build_component() {
  local name="$1" expected="$2" workers="$3"
  shift 3
  local output="$DATASET_DIRECTORY/$name"
  if aws s3 ls "$RUN_PREFIX/dataset/$name/dataset.json" >/dev/null 2>&1; then
    echo "skip existing dataset component $name"
    return
  fi
  "$VENV_DIRECTORY/bin/python" -m distill.inspect_teacher \
    "$@" --expected-records "$expected" --expected-candidates 8 \
    >"$WORK_DIRECTORY/inspection-$name.json"
  "$VENV_DIRECTORY/bin/python" -m distill.build_dataset \
    "$@" --output "$output" --records-per-shard 10_000 --workers "$workers"
  aws s3 sync "$output/" "$RUN_PREFIX/dataset/$name/" --only-show-errors
  aws s3 cp "$WORK_DIRECTORY/inspection-$name.json" \
    "$RUN_PREFIX/dataset/$name/inspection.json" --only-show-errors
}

build_component phase-c-extra-2m 2000000 8 "${ALL_INPUTS[@]:0:BASE_SHARDS}" &
FIRST_PID=$!
build_component phase-c-extra-7m 7000000 24 \
  "${ALL_INPUTS[@]:BASE_SHARDS:TOTAL_SHARDS-BASE_SHARDS}" &
SECOND_PID=$!
failed=0
if ! wait "$FIRST_PID"; then
  failed=1
fi
if ! wait "$SECOND_PID"; then
  failed=1
fi
if (( failed )); then
  echo "one or more Phase C dataset builds failed" >&2
  exit 1
fi

GIT_REVISION="$(git -c safe.directory="$REPOSITORY" rev-parse HEAD)"
jq -n --arg run_id "$RUN_ID" --arg base_run_id "$BASE_RUN_ID" \
  --arg git_revision "$GIT_REVISION" \
  '{run_id:$run_id,base_run_id:$base_run_id,status:"complete",git_revision:$git_revision}' \
  >"$WORK_DIRECTORY/status.json"
aws s3 cp "$WORK_DIRECTORY/status.json" "$RUN_PREFIX/status/phase-c-prepare.json" \
  --only-show-errors

if [[ "${DISTILL_SHUTDOWN:-1}" == "1" ]]; then
  sudo shutdown -h +1
fi
