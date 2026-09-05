#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
DATASET_NAME="${DISTILL_POLICY_DATASET:-phase-m-policy-1m}"
BASE_MODEL="${DISTILL_BASE_MODEL:-phase-e-c40-h64-frozen}"
MODEL_NAME="${DISTILL_MODEL_NAME:-phase-m-h64-metadata-1m}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
PYTORCH_PYTHON="${PYTORCH_PYTHON:-/opt/pytorch/bin/python}"
WORK_ROOT="${DISTILL_WORK_ROOT:-/home/ec2-user/aichessathon-work}"
WORK_DIRECTORY="${WORK_ROOT%/}/${MODEL_NAME}"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
DATASET_DIRECTORY="$WORK_DIRECTORY/dataset"
BASE_PATH="$WORK_DIRECTORY/${BASE_MODEL}.npz"
OUTPUT_PATH="$WORK_DIRECTORY/${MODEL_NAME}.npz"

schedule_shutdown() {
  if [[ "${DISTILL_SHUTDOWN:-1}" == "1" ]]; then
    sudo shutdown -h +1
  fi
}
trap schedule_shutdown EXIT

mkdir -p "$DATASET_DIRECTORY"
if [[ ! -d .deps-aws-distilled/chess ]]; then
  "$PYTORCH_PYTHON" -m pip install --target .deps-aws-distilled \
    -r training/requirements-distilled-aws.txt
fi
export PYTHONPATH="$PWD/.deps-aws-distilled${PYTHONPATH:+:$PYTHONPATH}"
aws s3 sync "$RUN_PREFIX/dataset/$DATASET_NAME/" "$DATASET_DIRECTORY/" \
  --only-show-errors
aws s3 cp "$RUN_PREFIX/models/$BASE_MODEL/$BASE_MODEL.npz" "$BASE_PATH" \
  --only-show-errors

"$PYTORCH_PYTHON" -m training.train_distilled \
  --data "$DATASET_DIRECTORY" \
  --output "$OUTPUT_PATH" \
  --device cuda \
  --epochs "${DISTILL_EPOCHS:-5}" \
  --batch-size "${DISTILL_BATCH_SIZE:-4096}" \
  --learning-rate "${DISTILL_LEARNING_RATE:-0.003}" \
  --accumulator 1024 --hidden 64 --bottleneck 32 \
  --initialize-from "$BASE_PATH" \
  --freeze-feature --freeze-base-head --metadata \
  --ranking-weight 0.5 --top-move-weight 0.75 \
  --antisymmetry-weight 0.5 --top-k 3 --top-k-ranking-boost 4 \
  --seed 31 | tee "$WORK_DIRECTORY/training.log"

aws s3 cp "$OUTPUT_PATH" "$RUN_PREFIX/models/$MODEL_NAME/$MODEL_NAME.npz" \
  --only-show-errors
aws s3 cp "${OUTPUT_PATH%.npz}.json" "$RUN_PREFIX/models/$MODEL_NAME/$MODEL_NAME.json" \
  --only-show-errors
aws s3 cp "$WORK_DIRECTORY/training.log" "$RUN_PREFIX/models/$MODEL_NAME/training.log" \
  --only-show-errors
jq -n --arg run_id "$RUN_ID" --arg model "$MODEL_NAME" \
  '{run_id:$run_id,model:$model,status:"complete"}' >"$WORK_DIRECTORY/status.json"
aws s3 cp "$WORK_DIRECTORY/status.json" "$RUN_PREFIX/status/training-${MODEL_NAME}.json" \
  --only-show-errors
