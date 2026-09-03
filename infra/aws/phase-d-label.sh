#!/usr/bin/env bash
set -euo pipefail

WORKER_INDEX="${1:?worker index is required}"
RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
WORKER_COUNT="${TEACHER_WORKER_COUNT:-30}"
SHARDS="${TEACHER_SHARDS:-22500}"
NODES="${TEACHER_NODES:-100000}"
PARALLELISM="${TEACHER_PARALLELISM:-32}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
WAIT_SECONDS="${PHASE_D_WAIT_SECONDS:-30}"
CORPUS_WAIT_LIMIT="${PHASE_D_CORPUS_WAIT_LIMIT:-480}"
LABEL_WAIT_LIMIT="${PHASE_D_LABEL_WAIT_LIMIT:-480}"

if ! [[ "$WORKER_INDEX" =~ ^[0-9]+$ ]] || (( WORKER_INDEX >= WORKER_COUNT )); then
  echo "worker index must be between zero and WORKER_COUNT - 1" >&2
  exit 2
fi

for ((attempt = 1; attempt <= CORPUS_WAIT_LIMIT; attempt++)); do
  if aws s3 ls "$RUN_PREFIX/corpus/manifest.json" >/dev/null 2>&1; then
    break
  fi
  if (( attempt == CORPUS_WAIT_LIMIT )); then
    echo "timed out waiting for the Phase D corpus" >&2
    sudo shutdown -h +1
    exit 1
  fi
  sleep "$WAIT_SECONDS"
done

worker_failed=0
TEACHER_RUN_ID="$RUN_ID" \
TEACHER_TIER=medium \
TEACHER_NODES="$NODES" \
TEACHER_SHARDS="$SHARDS" \
TEACHER_WORKER_INDEX="$WORKER_INDEX" \
TEACHER_WORKER_COUNT="$WORKER_COUNT" \
TEACHER_PARALLELISM="$PARALLELISM" \
TEACHER_SHUTDOWN=0 \
  bash infra/aws/teacher-worker.sh || worker_failed=$?

if (( worker_failed )); then
  sudo shutdown -h +1
  exit "$worker_failed"
fi
if (( WORKER_INDEX != 0 )); then
  sudo shutdown -h +1
  exit 0
fi

for ((attempt = 1; attempt <= LABEL_WAIT_LIMIT; attempt++)); do
  completed="$(aws s3 ls "$RUN_PREFIX/raw/medium/" --recursive | wc -l)"
  echo "Phase D label progress: $completed / $SHARDS raw shards"
  if (( completed == SHARDS )); then
    DISTILL_RUN_ID="$RUN_ID" bash infra/aws/phase-d-prepare.sh
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
