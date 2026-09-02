#!/usr/bin/env bash
set -euo pipefail

WORKER_INDEX="${1:?worker index from 0 through 7 is required}"
RUN_ID="${DISTILL_RUN_ID:-phase-c-20260902a}"
WORKER_COUNT="${TEACHER_WORKER_COUNT:-8}"
SHARDS="${TEACHER_SHARDS:-2250}"
NODES="${TEACHER_NODES:-100000}"
PARALLELISM="${TEACHER_PARALLELISM:-32}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
WAIT_SECONDS="${PHASE_C_WAIT_SECONDS:-60}"
WAIT_LIMIT="${PHASE_C_WAIT_LIMIT:-240}"

if ! [[ "$WORKER_INDEX" =~ ^[0-9]+$ ]] || (( WORKER_INDEX >= WORKER_COUNT )); then
  echo "worker index must be between zero and WORKER_COUNT - 1" >&2
  exit 2
fi

TEACHER_RUN_ID="$RUN_ID" \
TEACHER_TIER=medium \
TEACHER_NODES="$NODES" \
TEACHER_SHARDS="$SHARDS" \
TEACHER_WORKER_INDEX="$WORKER_INDEX" \
TEACHER_WORKER_COUNT="$WORKER_COUNT" \
TEACHER_PARALLELISM="$PARALLELISM" \
TEACHER_SHUTDOWN=0 \
  bash infra/aws/teacher-worker.sh

if (( WORKER_INDEX != 0 )); then
  sudo shutdown -h +1
  exit 0
fi

for ((attempt = 1; attempt <= WAIT_LIMIT; attempt++)); do
  completed="$(aws s3 ls "$RUN_PREFIX/raw/medium/" --recursive | wc -l)"
  echo "Phase C label progress: $completed / $SHARDS raw shards"
  if (( completed == SHARDS )); then
    DISTILL_RUN_ID="$RUN_ID" bash infra/aws/phase-c-prepare.sh
    exit 0
  fi
  if (( completed > SHARDS )); then
    echo "unexpected extra raw shards: found $completed, expected $SHARDS" >&2
    exit 1
  fi
  sleep "$WAIT_SECONDS"
done

echo "timed out waiting for all $SHARDS raw shards" >&2
exit 1
