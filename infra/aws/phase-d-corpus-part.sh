#!/usr/bin/env bash
set -euo pipefail

WORKER_INDEX="${1:?worker index is required}"
RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
WORKER_COUNT="${TEACHER_WORKER_COUNT:-30}"
POSITIONS="${DISTILL_PART_POSITIONS:-3000000}"
LOCAL_SHARDS="${DISTILL_PART_SHARDS:-750}"
SOURCES_PER_WORKER="${DISTILL_SOURCES_PER_WORKER:-4}"
FIRST_SOURCE="${DISTILL_FIRST_SOURCE_SHARD:-13}"
SEED_BASE="${DISTILL_SEED:-20260904}"
EXCLUDE_RUNS="${DISTILL_EXCLUDE_RUNS:-pilot-20260831a eval-20260901a scale-20260901a phase-c-20260902a}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
REPOSITORY="${AICHESSATHON_REPOSITORY:-/home/ec2-user/aichessathon}"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
WORK_DIRECTORY="/tmp/aichessathon-phase-d-corpus-${RUN_ID}-${WORKER_INDEX}"
VENV_DIRECTORY="$WORK_DIRECTORY/venv"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
CORPUS_DIRECTORY="$WORK_DIRECTORY/corpus"
GLOBAL_DIRECTORY="$WORK_DIRECTORY/global"
EXCLUDE_DIRECTORY="$WORK_DIRECTORY/exclude"

if ! [[ "$WORKER_INDEX" =~ ^[0-9]+$ ]] || (( WORKER_INDEX >= WORKER_COUNT )); then
  echo "worker index must be between zero and WORKER_COUNT - 1" >&2
  exit 2
fi
if (( POSITIONS % LOCAL_SHARDS != 0 )); then
  echo "DISTILL_PART_POSITIONS must divide evenly by DISTILL_PART_SHARDS" >&2
  exit 2
fi
STATUS_URI="$RUN_PREFIX/status/corpus/worker-${WORKER_INDEX}.json"
if aws s3 ls "$STATUS_URI" >/dev/null 2>&1; then
  echo "skip existing Phase D corpus partition $WORKER_INDEX"
  exit 0
fi

mkdir -p "$CORPUS_DIRECTORY" "$GLOBAL_DIRECTORY" "$EXCLUDE_DIRECTORY"
cd "$REPOSITORY"
if [[ ! -x "$VENV_DIRECTORY/bin/python" ]]; then
  "$UV_BIN" venv --python 3.12 "$VENV_DIRECTORY"
fi
if ! "$VENV_DIRECTORY/bin/python" -c 'import chess, pyarrow' >/dev/null 2>&1; then
  "$UV_BIN" pip install --python "$VENV_DIRECTORY/bin/python" \
    -r training/requirements-teacher-aws.txt \
    -r training/requirements-phase-b-aws.txt \
    pyarrow==25.0.1
fi

for prior_run in $EXCLUDE_RUNS; do
  if [[ ! "$prior_run" =~ ^[a-zA-Z0-9-]+$ ]]; then
    echo "DISTILL_EXCLUDE_RUNS contains an unsafe run name: $prior_run" >&2
    exit 2
  fi
  aws s3 sync "${ARTIFACTS_URI%/}/teacher/runs/$prior_run/corpus/" \
    "$EXCLUDE_DIRECTORY/$prior_run/" --exclude "*" --include "*.epd" --only-show-errors
done
mapfile -t EXCLUDE_PATHS < <(find "$EXCLUDE_DIRECTORY" -type f -name '*.epd' | sort)
if (( ${#EXCLUDE_PATHS[@]} == 0 )); then
  echo "no exclusion corpus files were found" >&2
  exit 1
fi

SOURCE_SHARDS=()
source_start=$((FIRST_SOURCE + WORKER_INDEX * SOURCES_PER_WORKER))
for ((offset = 0; offset < SOURCES_PER_WORKER; offset++)); do
  SOURCE_SHARDS+=("$((source_start + offset))")
done

"$VENV_DIRECTORY/bin/python" -m distill.sample_gigafish \
  --output "$CORPUS_DIRECTORY" \
  --cache "$WORK_DIRECTORY/cache" \
  --medium-positions "$POSITIONS" \
  --deep-positions 0 \
  --shards-per-tier "$LOCAL_SHARDS" \
  --source-shards "${SOURCE_SHARDS[@]}" \
  --seed "$((SEED_BASE + WORKER_INDEX))" \
  --exclude "${EXCLUDE_PATHS[@]}"

for ((local_shard = 0; local_shard < LOCAL_SHARDS; local_shard++)); do
  local_path="$(printf '%s/medium/part-%05d.epd' "$CORPUS_DIRECTORY" "$local_shard")"
  global_shard=$((WORKER_INDEX + local_shard * WORKER_COUNT))
  global_path="$(printf '%s/part-%05d.epd' "$GLOBAL_DIRECTORY" "$global_shard")"
  mv "$local_path" "$global_path"
done
aws s3 sync "$GLOBAL_DIRECTORY/" "$RUN_PREFIX/corpus/medium/" --only-show-errors
aws s3 cp "$CORPUS_DIRECTORY/manifest.json" \
  "$RUN_PREFIX/corpus/parts/worker-${WORKER_INDEX}-manifest.json" --only-show-errors

GIT_REVISION="$(git -c safe.directory="$REPOSITORY" rev-parse HEAD)"
jq -n --arg run_id "$RUN_ID" --arg git_revision "$GIT_REVISION" \
  --argjson worker_index "$WORKER_INDEX" --argjson worker_count "$WORKER_COUNT" \
  --argjson positions "$POSITIONS" --argjson shards "$LOCAL_SHARDS" \
  --argjson source_start "$source_start" --argjson sources "$SOURCES_PER_WORKER" \
  '{run_id:$run_id,status:"complete",git_revision:$git_revision,worker_index:$worker_index,worker_count:$worker_count,positions:$positions,shards:$shards,source_start:$source_start,source_count:$sources}' \
  >"$WORK_DIRECTORY/status.json"
aws s3 cp "$WORK_DIRECTORY/status.json" "$STATUS_URI" --only-show-errors
