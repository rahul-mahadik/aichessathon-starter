#!/usr/bin/env bash
set -euo pipefail

PYTORCH_PYTHON="${PYTORCH_PYTHON:-/opt/pytorch/bin/python}"
TRAINING_VENV="${TRAINING_VENV:-.venv-aws-training}"
DATA_PATH="${DISTILL_DATA:-training/data/distilled}"
OUTPUT_PATH="${DISTILL_OUTPUT:-weights/nnue.npz}"

if [[ ! -x "$PYTORCH_PYTHON" ]]; then
  echo "PyTorch DLAMI interpreter not found at $PYTORCH_PYTHON" >&2
  exit 1
fi

nvidia-smi
if [[ ! -x "$TRAINING_VENV/bin/python" ]]; then
  "$PYTORCH_PYTHON" -m venv --system-site-packages "$TRAINING_VENV"
  "$TRAINING_VENV/bin/python" -m pip install --upgrade pip
  "$TRAINING_VENV/bin/python" -m pip install -r training/requirements-aws.txt
fi

"$TRAINING_VENV/bin/python" training/device_check.py --require-cuda
"$TRAINING_VENV/bin/python" -m training.train_distilled \
  --device cuda \
  --data "$DATA_PATH" \
  --output "$OUTPUT_PATH" \
  "$@"

if [[ -n "${AICHESSATHON_ARTIFACTS_URI:-}" ]]; then
  aws s3 cp "$OUTPUT_PATH" \
    "${AICHESSATHON_ARTIFACTS_URI%/}/models/$(basename "$OUTPUT_PATH")"
  aws s3 cp "${OUTPUT_PATH%.npz}.json" \
    "${AICHESSATHON_ARTIFACTS_URI%/}/models/$(basename "${OUTPUT_PATH%.npz}.json")"
fi

