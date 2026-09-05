#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
DATASET_NAME="${DISTILL_POLICY_DATASET:-phase-m-policy-1m}"
EXPECTED_SHARDS="${DISTILL_POLICY_SHARDS:-250}"
EXPECTED_RECORDS="${DISTILL_EXPECTED_RECORDS:-1000000}"
BUILD_WORKERS="${DISTILL_BUILD_WORKERS:-24}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
REPOSITORY="${AICHESSATHON_REPOSITORY:-/home/ec2-user/aichessathon}"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
WORK_DIRECTORY="/tmp/aichessathon-${DATASET_NAME}"
VENV_DIRECTORY="$WORK_DIRECTORY/venv"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
RAW_DIRECTORY="$WORK_DIRECTORY/raw"
DATASET_DIRECTORY="$WORK_DIRECTORY/dataset"

schedule_shutdown() {
  if [[ "${DISTILL_SHUTDOWN:-1}" == "1" ]]; then
    sudo shutdown -h +1
  fi
}
trap schedule_shutdown EXIT

mkdir -p "$RAW_DIRECTORY" "$DATASET_DIRECTORY"
cd "$REPOSITORY"
if [[ ! -x "$VENV_DIRECTORY/bin/python" ]]; then
  "$UV_BIN" venv --python 3.12 "$VENV_DIRECTORY"
fi
if ! "$VENV_DIRECTORY/bin/python" -c 'import chess, numpy' >/dev/null 2>&1; then
  "$UV_BIN" pip install --python "$VENV_DIRECTORY/bin/python" \
    -r training/requirements-phase-b-aws.txt
fi

# The Phase D raw shards contain 4,000 records each. The first 250 make a
# deterministic one-million-position pilot without redownloading the full 90M.
aws s3 sync "$RUN_PREFIX/raw/medium/" "$RAW_DIRECTORY/" \
  --exclude "*" \
  --include "part-000??.jsonl.gz" \
  --include "part-001??.jsonl.gz" \
  --include "part-002[0-4]?.jsonl.gz" \
  --only-show-errors
mapfile -t RAW_INPUTS < <(find "$RAW_DIRECTORY" -type f -name '*.jsonl.gz' | sort)
if (( ${#RAW_INPUTS[@]} != EXPECTED_SHARDS )); then
  echo "found ${#RAW_INPUTS[@]} policy raw shards, expected $EXPECTED_SHARDS" >&2
  exit 1
fi

"$VENV_DIRECTORY/bin/python" -m distill.inspect_teacher \
  "${RAW_INPUTS[@]}" --expected-records "$EXPECTED_RECORDS" --expected-candidates 8 \
  >"$WORK_DIRECTORY/inspection.json"
"$VENV_DIRECTORY/bin/python" -m distill.build_dataset \
  "${RAW_INPUTS[@]}" --output "$DATASET_DIRECTORY" \
  --records-per-shard 10_000 --workers "$BUILD_WORKERS"
aws s3 sync "$DATASET_DIRECTORY/" "$RUN_PREFIX/dataset/$DATASET_NAME/" \
  --only-show-errors
aws s3 cp "$WORK_DIRECTORY/inspection.json" \
  "$RUN_PREFIX/dataset/$DATASET_NAME/inspection.json" --only-show-errors

jq -n --arg run_id "$RUN_ID" --arg dataset "$DATASET_NAME" \
  --argjson records "$EXPECTED_RECORDS" \
  '{run_id:$run_id,dataset:$dataset,records:$records,status:"complete"}' \
  >"$WORK_DIRECTORY/status.json"
aws s3 cp "$WORK_DIRECTORY/status.json" \
  "$RUN_PREFIX/status/${DATASET_NAME}.json" --only-show-errors
