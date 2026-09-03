#!/usr/bin/env bash
set -euo pipefail

CELL="${1:?cell required: D100, D100C20, or D100C40}"
RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
DATASETS="phase-c-base-1m phase-c-extra-2m phase-c-extra-7m phase-d-extra-90m"
EXPECTED_RECORDS=100000000

case "$CELL" in
  D100)
    MODEL_NAME=phase-d-d100m-c4p5
    ARCHITECTURE=(--accumulator 128 --hidden 64 --bottleneck 32)
    ;;
  D100C20)
    MODEL_NAME=phase-d-d100m-c20
    ARCHITECTURE=(--accumulator 512 --hidden 256 --bottleneck 128)
    ;;
  D100C40)
    MODEL_NAME=phase-d-d100m-c40
    ARCHITECTURE=(--accumulator 1024 --hidden 512 --bottleneck 256)
    ;;
  *)
    echo "cell must be D100, D100C20, or D100C40" >&2
    exit 2
    ;;
esac

DISTILL_RUN_ID="$RUN_ID" \
DISTILL_MODEL_NAME="$MODEL_NAME" \
DISTILL_REUSE_DATASET_MODELS="$DATASETS" \
DISTILL_EXPECTED_RECORDS="$EXPECTED_RECORDS" \
  bash infra/aws/distill-gpu-run.sh \
    --epochs "${DISTILL_EPOCHS:-5}" \
    --batch-size "${DISTILL_BATCH_SIZE:-4096}" \
    --learning-rate 0.001 \
    "${ARCHITECTURE[@]}" \
    --ranking-weight 0.5 \
    --top-move-weight 0.75 \
    --antisymmetry-weight 0.5 \
    --top-k 3 \
    --top-k-ranking-boost 4 \
    --seed 7
