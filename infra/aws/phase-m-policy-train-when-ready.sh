#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
DATASET_NAME="${DISTILL_POLICY_DATASET:-phase-m-policy-1m}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
WAIT_ATTEMPTS="${DISTILL_WAIT_ATTEMPTS:-240}"

schedule_shutdown() {
  if [[ "${DISTILL_SHUTDOWN:-1}" == "1" ]]; then
    sudo shutdown -h +1
  fi
}
trap schedule_shutdown EXIT

for ((attempt = 1; attempt <= WAIT_ATTEMPTS; attempt++)); do
  if aws s3 ls "$RUN_PREFIX/status/${DATASET_NAME}.json" >/dev/null 2>&1; then
    DISTILL_SHUTDOWN=0 bash infra/aws/phase-m-policy-train.sh
    exit 0
  fi
  echo "waiting for $DATASET_NAME ($attempt/$WAIT_ATTEMPTS)"
  sleep 30
done

echo "policy dataset did not become ready" >&2
exit 1
