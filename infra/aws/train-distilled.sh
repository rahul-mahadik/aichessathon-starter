#!/usr/bin/env bash
set -euo pipefail

PYTORCH_PYTHON="${PYTORCH_PYTHON:-/opt/pytorch/bin/python}"
TRAINING_DEPS="${TRAINING_DEPS:-.deps-aws-distilled}"
DATA_SPEC="${DISTILL_DATASETS:-${DISTILL_DATA:-training/data/distilled}}"
OUTPUT_PATH="${DISTILL_OUTPUT:-weights/nnue.npz}"

if [[ ! -x "$PYTORCH_PYTHON" ]]; then
  echo "PyTorch DLAMI interpreter not found at $PYTORCH_PYTHON" >&2
  exit 1
fi

nvidia-smi
if [[ ! -d "$TRAINING_DEPS/chess" ]]; then
  "$PYTORCH_PYTHON" -m pip install \
    --target "$TRAINING_DEPS" \
    -r training/requirements-distilled-aws.txt
fi
export PYTHONPATH="$PWD/$TRAINING_DEPS${PYTHONPATH:+:$PYTHONPATH}"
read -r -a DATA_PATHS <<<"$DATA_SPEC"
if (( ${#DATA_PATHS[@]} == 0 )); then
  echo "DISTILL_DATASETS must contain at least one dataset path" >&2
  exit 2
fi

"$PYTORCH_PYTHON" training/device_check.py --require-cuda
"$PYTORCH_PYTHON" -m training.train_distilled \
  --device cuda \
  --data "${DATA_PATHS[@]}" \
  --output "$OUTPUT_PATH" \
  "$@"

if [[ -n "${AICHESSATHON_ARTIFACTS_URI:-}" ]]; then
  ARTIFACT_PREFIX="${DISTILL_ARTIFACT_PREFIX:-${AICHESSATHON_ARTIFACTS_URI%/}/models}"
  aws s3 cp "$OUTPUT_PATH" \
    "${ARTIFACT_PREFIX%/}/$(basename "$OUTPUT_PATH")"
  aws s3 cp "${OUTPUT_PATH%.npz}.json" \
    "${ARTIFACT_PREFIX%/}/$(basename "${OUTPUT_PATH%.npz}.json")"
fi
