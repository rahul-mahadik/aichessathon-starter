#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
POLICY_MODEL="${DISTILL_POLICY_MODEL:-phase-m-h64-policy-1m}"
METADATA_MODEL="${DISTILL_METADATA_MODEL:-phase-m-h64-metadata-1m}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
WAIT_ATTEMPTS="${DISTILL_WAIT_ATTEMPTS:-360}"

shutdown_host() {
  sudo shutdown -h +1
}
trap shutdown_host EXIT

wait_for_model() {
  local model="$1"
  for ((attempt = 1; attempt <= WAIT_ATTEMPTS; attempt++)); do
    if aws s3 ls "$RUN_PREFIX/status/training-${model}.json" >/dev/null 2>&1; then
      return
    fi
    echo "waiting for $model ($attempt/$WAIT_ATTEMPTS)"
    sleep 30
  done
  echo "$model did not become ready" >&2
  exit 1
}

wait_for_model "$POLICY_MODEL"
BENCH_SHUTDOWN=0 bash infra/aws/phase-m-policy-benchmark.sh "$POLICY_MODEL"
wait_for_model "$METADATA_MODEL"
BENCH_SHUTDOWN=0 bash infra/aws/phase-m-metadata-benchmark.sh "$METADATA_MODEL"
