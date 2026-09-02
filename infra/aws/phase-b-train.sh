#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model required: M, R-deep, H-medium, or H-deep}"
RUN_ID="${DISTILL_RUN_ID:-scale-20260901a}"

case "$MODEL" in
  M)
    MODEL_NAME=phase-b-m
    DATASETS=phase-b-m-base
    EXPECTED_RECORDS=1000000
    ;;
  R-deep)
    MODEL_NAME=phase-b-r-deep
    DATASETS="phase-b-m-base phase-b-r-deep-extra"
    EXPECTED_RECORDS=1100000
    ;;
  H-medium)
    MODEL_NAME=phase-b-h-medium
    DATASETS="phase-b-m-base phase-b-h-medium-extra"
    EXPECTED_RECORDS=1100000
    ;;
  H-deep)
    MODEL_NAME=phase-b-h-deep
    DATASETS="phase-b-m-base phase-b-h-deep-extra"
    EXPECTED_RECORDS=1100000
    ;;
  *)
    echo "model must be M, R-deep, H-medium, or H-deep" >&2
    exit 2
    ;;
esac

DISTILL_RUN_ID="$RUN_ID" \
DISTILL_MODEL_NAME="$MODEL_NAME" \
DISTILL_REUSE_DATASET_MODELS="$DATASETS" \
DISTILL_EXPECTED_RECORDS="$EXPECTED_RECORDS" \
  bash infra/aws/distill-gpu-run.sh \
    --epochs 20 \
    --batch-size 2048 \
    --max-train-batches 500 \
    --learning-rate 0.001 \
    --accumulator 128 \
    --hidden 64 \
    --bottleneck 32 \
    --ranking-weight 0.5 \
    --top-move-weight 0.75 \
    --antisymmetry-weight 0.5 \
    --top-k 3 \
    --top-k-ranking-boost 4 \
    --seed 7
