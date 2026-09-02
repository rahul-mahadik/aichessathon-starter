#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:-scale-20260901a}"
SELECT="${DISTILL_SELECT:-100000}"
HOLDOUT="${DISTILL_HOLDOUT:-20000}"
SEED="${DISTILL_SEED:-7}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
REPOSITORY="${AICHESSATHON_REPOSITORY:-/home/ec2-user/aichessathon}"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
WORK_DIRECTORY="/tmp/aichessathon-phase-b-${RUN_ID}"
VENV_DIRECTORY="$WORK_DIRECTORY/venv"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
RAW_DIRECTORY="$WORK_DIRECTORY/raw"
MINED_DIRECTORY="$WORK_DIRECTORY/mined"
DATASET_DIRECTORY="$WORK_DIRECTORY/dataset"

if ! [[ "$SELECT" =~ ^[1-9][0-9]*$ && "$HOLDOUT" =~ ^[0-9]+$ && "$SEED" =~ ^[0-9]+$ ]]; then
  echo "DISTILL_SELECT must be positive; DISTILL_HOLDOUT and DISTILL_SEED must be non-negative" >&2
  exit 2
fi

mkdir -p "$RAW_DIRECTORY" "$MINED_DIRECTORY" "$DATASET_DIRECTORY"
cd "$REPOSITORY"
if [[ ! -x "$VENV_DIRECTORY/bin/python" ]]; then
  "$UV_BIN" venv --python 3.12 "$VENV_DIRECTORY"
fi
if ! "$VENV_DIRECTORY/bin/python" -c 'import chess, numpy' >/dev/null 2>&1; then
  "$UV_BIN" pip install --python "$VENV_DIRECTORY/bin/python" \
    -r training/requirements-phase-b-aws.txt
fi

download_tier() {
  local tier="$1"
  mkdir -p "$RAW_DIRECTORY/$tier"
  aws s3 sync "$RUN_PREFIX/raw/$tier/" "$RAW_DIRECTORY/$tier/" \
    --exclude "*" --include "*.jsonl.gz" --only-show-errors
}

for tier in medium deep-10k deep-100k deep-1m; do
  download_tier "$tier"
done

mapfile -t LOW_INPUTS < <(find "$RAW_DIRECTORY/deep-10k" -type f -name '*.jsonl.gz' | sort)
mapfile -t MEDIUM_INPUTS < <(find "$RAW_DIRECTORY/deep-100k" -type f -name '*.jsonl.gz' | sort)
mapfile -t DEEP_INPUTS < <(find "$RAW_DIRECTORY/deep-1m" -type f -name '*.jsonl.gz' | sort)

"$VENV_DIRECTORY/bin/python" -m distill.mine_depth_disagreements \
  --low "${LOW_INPUTS[@]}" \
  --medium "${MEDIUM_INPUTS[@]}" \
  --deep "${DEEP_INPUTS[@]}" \
  --select "$SELECT" \
  --holdout "$HOLDOUT" \
  --seed "$SEED" \
  --output "$MINED_DIRECTORY"
aws s3 sync "$MINED_DIRECTORY/" "$RUN_PREFIX/mined/phase-b-ablation/" --only-show-errors

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
    >"$MINED_DIRECTORY/inspection-$name.json"
  "$VENV_DIRECTORY/bin/python" -m distill.build_dataset \
    "$@" --output "$output" --records-per-shard 10_000 --workers "$workers"
  aws s3 sync "$output/" "$RUN_PREFIX/dataset/$name/" --only-show-errors
  aws s3 cp "$MINED_DIRECTORY/inspection-$name.json" \
    "$RUN_PREFIX/dataset/$name/inspection.json" --only-show-errors
}

mapfile -t BASE_INPUTS < <(find "$RAW_DIRECTORY/medium" -type f -name '*.jsonl.gz' | sort)
build_component phase-b-m-base 1000000 32 "${BASE_INPUTS[@]}"

pids=()
build_component phase-b-r-deep-extra "$SELECT" 1 "$MINED_DIRECTORY/random-deep.jsonl.gz" &
pids+=("$!")
build_component phase-b-h-medium-extra "$SELECT" 1 "$MINED_DIRECTORY/high-medium.jsonl.gz" &
pids+=("$!")
build_component phase-b-h-deep-extra "$SELECT" 1 "$MINED_DIRECTORY/high-deep.jsonl.gz" &
pids+=("$!")
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed )); then
  echo "one or more Phase B extra dataset builds failed" >&2
  exit 1
fi

STATUS_PATH="$WORK_DIRECTORY/status.json"
GIT_REVISION="$(git rev-parse HEAD)"
jq -n \
  --arg run_id "$RUN_ID" \
  --arg git_revision "$GIT_REVISION" \
  --argjson selected "$SELECT" \
  --argjson holdout "$HOLDOUT" \
  --argjson seed "$SEED" \
  '{run_id:$run_id,status:"complete",git_revision:$git_revision,selected:$selected,holdout:$holdout,seed:$seed}' \
  >"$STATUS_PATH"
aws s3 cp "$STATUS_PATH" "$RUN_PREFIX/status/phase-b-prepare.json" --only-show-errors

if [[ "${DISTILL_SHUTDOWN:-1}" == "1" ]]; then
  sudo shutdown -h +1
fi
