#!/usr/bin/env bash
set -euo pipefail

CELL="${1:?cell required: H256FROZEN, H128FROZEN, H64FROZEN, H32FROZEN, or H64REFINE}"
RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
WORK_ROOT="${DISTILL_WORK_ROOT:-/home/ec2-user/aichessathon-work}"
DATASETS="phase-c-base-1m phase-c-extra-2m phase-c-extra-7m phase-d-extra-90m"
EXPECTED_RECORDS=100000000

case "$CELL" in
  H256FROZEN)
    MODEL_NAME=phase-e-c40-h256-frozen
    TEACHER_NAME=phase-d-d100m-c40
    ARCHITECTURE=(--accumulator 1024 --hidden 256 --bottleneck 128)
    ;;
  H128FROZEN)
    MODEL_NAME=phase-e-c40-h128-frozen
    TEACHER_NAME=phase-d-d100m-c40
    ARCHITECTURE=(--accumulator 1024 --hidden 128 --bottleneck 64)
    ;;
  H64FROZEN)
    MODEL_NAME=phase-e-c40-h64-frozen
    TEACHER_NAME=phase-d-d100m-c40
    ARCHITECTURE=(--accumulator 1024 --hidden 64 --bottleneck 32)
    ;;
  H32FROZEN)
    MODEL_NAME=phase-e-c40-h32-frozen
    TEACHER_NAME=phase-d-d100m-c40
    ARCHITECTURE=(--accumulator 1024 --hidden 32 --bottleneck 32)
    ;;
  H64REFINE)
    MODEL_NAME=phase-e-c40-h64-refined
    TEACHER_NAME=phase-e-c40-h64-frozen
    ARCHITECTURE=(--accumulator 1024 --hidden 64 --bottleneck 32)
    ;;
  *)
    echo "unsupported Phase E cell: $CELL" >&2
    exit 2
    ;;
esac

TEACHER_PATH="${WORK_ROOT%/}/teachers/${TEACHER_NAME}.npz"

mkdir -p "$(dirname "$TEACHER_PATH")"
if [[ ! -f "$TEACHER_PATH" ]]; then
  aws s3 cp \
    "${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}/models/${TEACHER_NAME}/${TEACHER_NAME}.npz" \
    "$TEACHER_PATH" --only-show-errors
fi

DISTILL_RUN_ID="$RUN_ID" \
DISTILL_MODEL_NAME="$MODEL_NAME" \
DISTILL_REUSE_DATASET_MODELS="$DATASETS" \
DISTILL_EXPECTED_RECORDS="$EXPECTED_RECORDS" \
DISTILL_WORK_ROOT="$WORK_ROOT" \
DISTILL_DATASET_CACHE_ROOT="${WORK_ROOT%/}/dataset-cache/${RUN_ID}" \
  bash infra/aws/distill-gpu-run.sh \
    --epochs "${DISTILL_EPOCHS:-5}" \
    --batch-size "${DISTILL_BATCH_SIZE:-4096}" \
    --learning-rate "${DISTILL_LEARNING_RATE:-0.001}" \
    "${ARCHITECTURE[@]}" \
    --initialize-from "$TEACHER_PATH" \
    --freeze-feature \
    --ranking-weight 0.5 \
    --top-move-weight 0.75 \
    --antisymmetry-weight 0.5 \
    --top-k 3 \
    --top-k-ranking-boost 4 \
    --seed 7
