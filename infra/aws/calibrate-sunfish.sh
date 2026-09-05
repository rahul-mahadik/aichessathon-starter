#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${DISTILL_RUN_ID:-phase-d-20260903a}"
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
CANDIDATE="${CALIBRATION_AGENT:-controls/classical}"
OPPONENT="${CALIBRATION_OPPONENT:-controls/sunfish_dupe}"
POSITIONS="${CALIBRATION_POSITIONS:-benchmarks/calibration-openings.epd}"
ROUNDS="${CALIBRATION_ROUNDS:-1}"
WORKERS="${CALIBRATION_WORKERS:-16}"
BASE_MS="${CALIBRATION_BASE_MS:-120000}"
INCREMENT_MS="${CALIBRATION_INCREMENT_MS:-500}"
NAME="${CALIBRATION_NAME:-classical-vs-house-sunfish-clock}"
OUTPUT="/tmp/${NAME}.json"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

"$UV_BIN" sync --frozen --python 3.12
"$UV_BIN" run python -m benchmarks.run \
  --agent "$CANDIDATE" \
  --opponent "$OPPONENT" \
  --positions "$POSITIONS" \
  --rounds "$ROUNDS" \
  --base-ms "$BASE_MS" \
  --increment-ms "$INCREMENT_MS" \
  --workers "$WORKERS" \
  --output "$OUTPUT"

aws s3 cp "$OUTPUT" \
  "${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}/calibration/$(basename "$OUTPUT")" \
  --only-show-errors

if [[ "${CALIBRATION_SHUTDOWN:-1}" == "1" ]]; then
  sudo shutdown -h +1
fi
