#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:?DISTILL_RUN_ID is required}"
MODEL_NAME="${DISTILL_MODEL_NAME:-combined}"
TIERS="${DISTILL_TIERS:-medium deep}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
PYTORCH_PYTHON="${PYTORCH_PYTHON:-/opt/pytorch/bin/python}"
TRAINING_DEPS="${TRAINING_DEPS:-.deps-aws-distilled}"
WORK_DIRECTORY="/tmp/aichessathon-distill-${RUN_ID}-${MODEL_NAME}"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
RAW_DIRECTORY="$WORK_DIRECTORY/raw"
DATASET_DIRECTORY="$WORK_DIRECTORY/dataset"
OUTPUT_PATH="$WORK_DIRECTORY/${MODEL_NAME}.npz"

if [[ ! -x "$PYTORCH_PYTHON" ]]; then
  echo "PyTorch DLAMI interpreter not found at $PYTORCH_PYTHON" >&2
  exit 1
fi

mkdir -p "$RAW_DIRECTORY" "$DATASET_DIRECTORY"
if [[ ! -d "$TRAINING_DEPS/chess" ]]; then
  "$PYTORCH_PYTHON" -m pip install \
    --target "$TRAINING_DEPS" \
    -r training/requirements-distilled-aws.txt
fi
export PYTHONPATH="$PWD/$TRAINING_DEPS${PYTHONPATH:+:$PYTHONPATH}"

for tier in $TIERS; do
  case "$tier" in
    medium|deep) ;;
    *) echo "DISTILL_TIERS contains unsupported tier: $tier" >&2; exit 2 ;;
  esac
  mkdir -p "$RAW_DIRECTORY/$tier"
  aws s3 sync "$RUN_PREFIX/raw/$tier/" "$RAW_DIRECTORY/$tier/" \
    --exclude "*" --include "*.jsonl.gz"
done

mapfile -t RAW_INPUTS < <(find "$RAW_DIRECTORY" -type f -name '*.jsonl.gz' | sort)
if (( ${#RAW_INPUTS[@]} == 0 )); then
  echo "No raw teacher records found for $RUN_ID ($TIERS)" >&2
  exit 1
fi

"$PYTORCH_PYTHON" -m distill.build_dataset \
  "${RAW_INPUTS[@]}" \
  --output "$DATASET_DIRECTORY" \
  --records-per-shard 10_000
aws s3 sync "$DATASET_DIRECTORY/" "$RUN_PREFIX/dataset/$MODEL_NAME/"

DISTILL_DATA="$DATASET_DIRECTORY" \
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
