#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:-phase-c-20260902a}"
STATUS_NAME="${DISTILL_CORPUS_STATUS_NAME:-phase-c-corpus.json}"
MEDIUM_POSITIONS="${DISTILL_MEDIUM_POSITIONS:-9000000}"
SHARDS="${DISTILL_SHARDS:-2250}"
SEED="${DISTILL_SEED:-20260903}"
SOURCE_SHARDS="${DISTILL_SOURCE_SHARDS:-1 2 3 4 5 6 7 8 9 10 11 12}"
EXCLUDE_RUNS="${DISTILL_EXCLUDE_RUNS:-pilot-20260831a eval-20260901a scale-20260901a}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
REPOSITORY="${AICHESSATHON_REPOSITORY:-/home/ec2-user/aichessathon}"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
WORK_DIRECTORY="/tmp/aichessathon-phase-c-corpus-${RUN_ID}"
VENV_DIRECTORY="$WORK_DIRECTORY/venv"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
CORPUS_DIRECTORY="$WORK_DIRECTORY/corpus"
EXCLUDE_DIRECTORY="$WORK_DIRECTORY/exclude"

if ! [[ "$MEDIUM_POSITIONS" =~ ^[1-9][0-9]*$ && "$SHARDS" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^[0-9]+$ ]]; then
  echo "position, shard, and seed values must be positive integers" >&2
  exit 2
fi
if (( MEDIUM_POSITIONS % SHARDS != 0 )); then
  echo "DISTILL_MEDIUM_POSITIONS must divide evenly by DISTILL_SHARDS for exact nested components" >&2
  exit 2
fi
if aws s3 ls "$RUN_PREFIX/corpus/manifest.json" >/dev/null 2>&1; then
  echo "skip existing Phase C corpus: $RUN_PREFIX/corpus/manifest.json"
  exit 0
fi

mkdir -p "$CORPUS_DIRECTORY" "$EXCLUDE_DIRECTORY"
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
  aws s3 sync \
    "${ARTIFACTS_URI%/}/teacher/runs/$prior_run/corpus/" \
    "$EXCLUDE_DIRECTORY/$prior_run/" \
    --exclude "*" --include "*.epd" --only-show-errors
done
mapfile -t EXCLUDE_PATHS < <(find "$EXCLUDE_DIRECTORY" -type f -name '*.epd' | sort)
if (( ${#EXCLUDE_PATHS[@]} == 0 )); then
  echo "no exclusion corpus files were found" >&2
  exit 1
fi
read -r -a SOURCE_SHARD_IDS <<<"$SOURCE_SHARDS"

"$VENV_DIRECTORY/bin/python" -m distill.sample_gigafish \
  --output "$CORPUS_DIRECTORY" \
  --cache "$WORK_DIRECTORY/cache" \
  --medium-positions "$MEDIUM_POSITIONS" \
  --deep-positions 0 \
  --shards-per-tier "$SHARDS" \
  --source-shards "${SOURCE_SHARD_IDS[@]}" \
  --seed "$SEED" \
  --exclude "${EXCLUDE_PATHS[@]}"

aws s3 sync "$CORPUS_DIRECTORY/" "$RUN_PREFIX/corpus/" --only-show-errors
GIT_REVISION="$(git -c safe.directory="$REPOSITORY" rev-parse HEAD)"
jq -n \
  --arg run_id "$RUN_ID" \
  --arg git_revision "$GIT_REVISION" \
  --argjson positions "$MEDIUM_POSITIONS" \
  --argjson shards "$SHARDS" \
  --argjson seed "$SEED" \
  '{run_id:$run_id,status:"complete",git_revision:$git_revision,positions:$positions,shards:$shards,seed:$seed}' \
  >"$WORK_DIRECTORY/status.json"
aws s3 cp "$WORK_DIRECTORY/status.json" "$RUN_PREFIX/status/$STATUS_NAME" \
  --only-show-errors

if [[ "${DISTILL_SHUTDOWN:-1}" == "1" ]]; then
  sudo shutdown -h +1
fi
