#!/usr/bin/env bash
set -euo pipefail

CELL="${1:?cell required: D100, D100C20, or D100C40}"
RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
WAIT_SECONDS="${DISTILL_WAIT_SECONDS:-60}"
WAIT_LIMIT="${DISTILL_WAIT_LIMIT:-720}"
READY_URI="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}/dataset/phase-d-extra-90m/dataset.json"

for ((attempt = 1; attempt <= WAIT_LIMIT; attempt++)); do
  if aws s3 ls "$READY_URI" >/dev/null 2>&1; then
    echo "Phase D dataset ready; starting $CELL"
    DISTILL_RUN_ID="$RUN_ID" bash infra/aws/phase-d-train.sh "$CELL"
    exit 0
  fi
  if (( attempt % 10 == 0 )); then
    echo "Waiting for Phase D dataset: attempt $attempt / $WAIT_LIMIT"
  fi
  sleep "$WAIT_SECONDS"
done

echo "timed out waiting for $READY_URI" >&2
if [[ "${DISTILL_SHUTDOWN:-1}" == "1" ]]; then
  sudo shutdown -h +1
fi
exit 1
