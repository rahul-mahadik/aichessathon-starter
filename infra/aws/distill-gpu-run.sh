#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:?DISTILL_RUN_ID is required}"
MODEL_NAME="${DISTILL_MODEL_NAME:-combined}"
TIERS="${DISTILL_TIERS:-medium deep}"
REUSE_DATASET_MODEL="${DISTILL_REUSE_DATASET_MODEL:-}"
REUSE_DATASET_MODELS="${DISTILL_REUSE_DATASET_MODELS:-$REUSE_DATASET_MODEL}"
EXPECTED_RECORDS="${DISTILL_EXPECTED_RECORDS:?DISTILL_EXPECTED_RECORDS is required}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
PYTORCH_PYTHON="${PYTORCH_PYTHON:-/opt/pytorch/bin/python}"
TRAINING_DEPS="${TRAINING_DEPS:-.deps-aws-distilled}"
WORK_ROOT="${DISTILL_WORK_ROOT:-/tmp}"
DATASET_CACHE_ROOT="${DISTILL_DATASET_CACHE_ROOT:-}"
WORK_DIRECTORY="${WORK_ROOT%/}/aichessathon-distill-${RUN_ID}-${MODEL_NAME}"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
RAW_DIRECTORY="$WORK_DIRECTORY/raw"
DATASET_DIRECTORY="$WORK_DIRECTORY/dataset"
OUTPUT_PATH="$WORK_DIRECTORY/${MODEL_NAME}.npz"

if [[ ! -x "$PYTORCH_PYTHON" ]]; then
  echo "PyTorch DLAMI interpreter not found at $PYTORCH_PYTHON" >&2
  exit 1
fi

mkdir -p "$WORK_ROOT" "$RAW_DIRECTORY" "$DATASET_DIRECTORY"
if [[ ! -d "$TRAINING_DEPS/chess" ]]; then
  "$PYTORCH_PYTHON" -m pip install \
    --target "$TRAINING_DEPS" \
    -r training/requirements-distilled-aws.txt
fi
export PYTHONPATH="$PWD/$TRAINING_DEPS${PYTHONPATH:+:$PYTHONPATH}"

INSPECTION_PATH="$WORK_DIRECTORY/inspection.json"
DATASET_PATHS=()
if [[ -n "$REUSE_DATASET_MODELS" ]]; then
  DATASET_RECORDS=0
  for dataset_model in $REUSE_DATASET_MODELS; do
    if [[ ! "$dataset_model" =~ ^[a-zA-Z0-9-]+$ ]]; then
      echo "DISTILL_REUSE_DATASET_MODELS contains an unsafe name: $dataset_model" >&2
      exit 2
    fi
    if [[ -n "$DATASET_CACHE_ROOT" ]]; then
      component_directory="${DATASET_CACHE_ROOT%/}/$dataset_model"
    else
      component_directory="$DATASET_DIRECTORY/$dataset_model"
    fi
    mkdir -p "$component_directory"
    aws s3 sync "$RUN_PREFIX/dataset/$dataset_model/" "$component_directory/"
    if [[ ! -f "$component_directory/dataset.json" ]]; then
      echo "Reusable dataset $dataset_model has no dataset.json" >&2
      exit 1
    fi
    component_records="$(jq -r '.records' "$component_directory/dataset.json")"
    DATASET_RECORDS=$((DATASET_RECORDS + component_records))
    DATASET_PATHS+=("$component_directory")
  done
  if [[ "$DATASET_RECORDS" != "$EXPECTED_RECORDS" ]]; then
    echo "Reusable datasets have $DATASET_RECORDS records, expected $EXPECTED_RECORDS" >&2
    exit 1
  fi
else
  for tier in $TIERS; do
    if [[ ! "$tier" =~ ^[a-zA-Z0-9-]+$ ]]; then
      echo "DISTILL_TIERS contains an unsafe label: $tier" >&2
      exit 2
    fi
    mkdir -p "$RAW_DIRECTORY/$tier"
    aws s3 sync "$RUN_PREFIX/raw/$tier/" "$RAW_DIRECTORY/$tier/" \
      --exclude "*" --include "*.jsonl.gz"
  done

  mapfile -t RAW_INPUTS < <(find "$RAW_DIRECTORY" -type f -name '*.jsonl.gz' | sort)
  if (( ${#RAW_INPUTS[@]} == 0 )); then
    echo "No raw teacher records found for $RUN_ID ($TIERS)" >&2
    exit 1
  fi

  "$PYTORCH_PYTHON" -m distill.inspect_teacher \
    "${RAW_INPUTS[@]}" \
    --expected-records "$EXPECTED_RECORDS" \
    --expected-candidates 8 >"$INSPECTION_PATH"
  cat "$INSPECTION_PATH"
  aws s3 cp "$INSPECTION_PATH" "$RUN_PREFIX/dataset/$MODEL_NAME/inspection.json"

  "$PYTORCH_PYTHON" -m distill.build_dataset \
    "${RAW_INPUTS[@]}" \
    --output "$DATASET_DIRECTORY" \
    --records-per-shard 10_000
  aws s3 sync "$DATASET_DIRECTORY/" "$RUN_PREFIX/dataset/$MODEL_NAME/"
  DATASET_PATHS+=("$DATASET_DIRECTORY")
fi

DISTILL_DATASETS="${DATASET_PATHS[*]}" \
DISTILL_OUTPUT="$OUTPUT_PATH" \
DISTILL_ARTIFACT_PREFIX="$RUN_PREFIX/models/$MODEL_NAME" \
  bash infra/aws/train-distilled.sh "$@"

STATUS_PATH="$WORK_DIRECTORY/status.json"
printf '{"run_id":"%s","model":"%s","tiers":"%s","status":"complete"}\n' \
  "$RUN_ID" "$MODEL_NAME" "$TIERS" >"$STATUS_PATH"
aws s3 cp "$STATUS_PATH" "$RUN_PREFIX/status/training-${MODEL_NAME}.json"
if [[ "${DISTILL_SHUTDOWN:-1}" == "1" ]]; then
  sudo shutdown -h +1
fi
