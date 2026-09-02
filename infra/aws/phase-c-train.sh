#!/usr/bin/env bash
set -euo pipefail

CELL="${1:?cell required: D3, D10, C10, C20, or C40}"
RUN_ID="${DISTILL_RUN_ID:-phase-c-20260902a}"

case "$CELL" in
  D3)
    MODEL_NAME=phase-c-d3m-c4p5
    DATASETS="phase-c-base-1m phase-c-extra-2m"
    EXPECTED_RECORDS=3000000
    ARCHITECTURE=(--accumulator 128 --hidden 64 --bottleneck 32)
    ;;
  D10)
    MODEL_NAME=phase-c-d10m-c4p5
    DATASETS="phase-c-base-1m phase-c-extra-2m phase-c-extra-7m"
    EXPECTED_RECORDS=10000000
    ARCHITECTURE=(--accumulator 128 --hidden 64 --bottleneck 32)
    ;;
  C10)
    MODEL_NAME=phase-c-d1m-c10
    DATASETS=phase-c-base-1m
    EXPECTED_RECORDS=1000000
    ARCHITECTURE=(--accumulator 256 --hidden 128 --bottleneck 64)
    ;;
  C20)
    MODEL_NAME=phase-c-d1m-c20
    DATASETS=phase-c-base-1m
    EXPECTED_RECORDS=1000000
    ARCHITECTURE=(--accumulator 512 --hidden 256 --bottleneck 128)
    ;;
  C40)
    MODEL_NAME=phase-c-d1m-c40
    DATASETS=phase-c-base-1m
    EXPECTED_RECORDS=1000000
    ARCHITECTURE=(--accumulator 1024 --hidden 512 --bottleneck 256)
    ;;
  *)
    echo "cell must be D3, D10, C10, C20, or C40" >&2
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
    --learning-rate 0.001 \
    "${ARCHITECTURE[@]}" \
    --ranking-weight 0.5 \
    --top-move-weight 0.75 \
    --antisymmetry-weight 0.5 \
    --top-k 3 \
    --top-k-ranking-boost 4 \
    --seed 7
