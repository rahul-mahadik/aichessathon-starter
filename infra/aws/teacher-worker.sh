#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${TEACHER_RUN_ID:?TEACHER_RUN_ID is required}"
TIER="${TEACHER_TIER:?TEACHER_TIER is required}"
LABEL="${TEACHER_LABEL:-$TIER}"
NODE_BUDGET="${TEACHER_NODES:?TEACHER_NODES is required}"
SHARD_COUNT="${TEACHER_SHARDS:?TEACHER_SHARDS is required}"
WORKER_INDEX="${TEACHER_WORKER_INDEX:-0}"
WORKER_COUNT="${TEACHER_WORKER_COUNT:-1}"
PARALLELISM="${TEACHER_PARALLELISM:-$(nproc)}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
REPOSITORY="${AICHESSATHON_REPOSITORY:-$HOME/aichessathon}"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
STOCKFISH_BIN="${STOCKFISH_PATH:-/usr/local/bin/stockfish}"
WORK_DIRECTORY="/tmp/aichessathon-teacher-${RUN_ID}-${TIER}-${WORKER_INDEX}"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"

if [[ "$TIER" != "medium" && "$TIER" != "deep" ]]; then
  echo "TEACHER_TIER must be medium or deep" >&2
  exit 2
fi
if [[ ! "$LABEL" =~ ^[a-zA-Z0-9-]+$ ]]; then
  echo "TEACHER_LABEL may contain only letters, numbers, and hyphens" >&2
  exit 2
fi
if (( SHARD_COUNT < 1 || WORKER_COUNT < 1 || WORKER_INDEX < 0 || WORKER_INDEX >= WORKER_COUNT )); then
  echo "invalid shard or worker configuration" >&2
  exit 2
fi

mkdir -p "$WORK_DIRECTORY/input" "$WORK_DIRECTORY/output" "$WORK_DIRECTORY/logs"
cd "$REPOSITORY"
sudo bash infra/aws/install-stockfish.sh
"$UV_BIN" venv --python 3.12 "$WORK_DIRECTORY/venv"
"$UV_BIN" pip install --python "$WORK_DIRECTORY/venv/bin/python" \
  -r training/requirements-teacher-aws.txt

run_shard() {
  local shard_number="$1" shard_name input_path output_path log_path output_uri
  shard_name="$(printf 'part-%05d' "$shard_number")"
  input_path="$WORK_DIRECTORY/input/${shard_name}.epd"
  output_path="$WORK_DIRECTORY/output/${shard_name}.jsonl.gz"
  log_path="$WORK_DIRECTORY/logs/${shard_name}.log"
  output_uri="$RUN_PREFIX/raw/$LABEL/${shard_name}.jsonl.gz"

  if aws s3 ls "$output_uri" >/dev/null 2>&1; then
    echo "skip existing $output_uri"
    return
  fi
  aws s3 cp "$RUN_PREFIX/corpus/$TIER/${shard_name}.epd" "$input_path"
  if ! "$WORK_DIRECTORY/venv/bin/python" -m distill.annotate \
    --input "$input_path" \
    --output "$output_path" \
    --stockfish "$STOCKFISH_BIN" \
    --nodes "$NODE_BUDGET" \
    --multipv 8 \
    --progress-every 250 >"$log_path" 2>&1; then
    aws s3 cp "$log_path" "$RUN_PREFIX/logs/$LABEL/worker-${WORKER_INDEX}-${shard_name}.log"
    return 1
  fi
  aws s3 cp "$output_path" "$output_uri"
  aws s3 cp "$log_path" "$RUN_PREFIX/logs/$LABEL/worker-${WORKER_INDEX}-${shard_name}.log"
}

pids=()
failed=0
wait_group() {
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  pids=()
}

for ((shard = WORKER_INDEX; shard < SHARD_COUNT; shard += WORKER_COUNT)); do
  run_shard "$shard" &
  pids+=("$!")
  if (( ${#pids[@]} >= PARALLELISM )); then
    wait_group
  fi
done
if (( ${#pids[@]} > 0 )); then
  wait_group
fi

STATUS_PATH="$WORK_DIRECTORY/status.json"
printf '{"run_id":"%s","tier":"%s","label":"%s","worker_index":%d,"worker_count":%d,"failed":%d}\n' \
  "$RUN_ID" "$TIER" "$LABEL" "$WORKER_INDEX" "$WORKER_COUNT" "$failed" >"$STATUS_PATH"
aws s3 cp "$STATUS_PATH" "$RUN_PREFIX/status/$LABEL/worker-${WORKER_INDEX}.json"
if [[ "${TEACHER_SHUTDOWN:-1}" == "1" ]]; then
  sudo shutdown -h +1
fi
exit "$failed"
