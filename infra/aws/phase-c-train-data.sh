#!/usr/bin/env bash
set -uo pipefail

RUN_ID="${DISTILL_RUN_ID:-phase-c-20260902a}"
if [[ -f /etc/profile.d/aichessathon.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/aichessathon.sh
fi
ARTIFACTS_URI="${AICHESSATHON_ARTIFACTS_URI:?AICHESSATHON_ARTIFACTS_URI is required}"
LOG_DIRECTORY="/tmp/aichessathon-phase-c-data-training-${RUN_ID}"
RUN_PREFIX="${ARTIFACTS_URI%/}/teacher/runs/${RUN_ID}"
mkdir -p "$LOG_DIRECTORY"

schedule_shutdown() {
  sudo shutdown -c >/dev/null 2>&1 || true
  sudo shutdown -h +1 >/dev/null 2>&1 || true
}
trap schedule_shutdown EXIT

# A fresh boot clears the launch-time shutdown timer, so restore a hard six-hour ceiling before
# starting either process. The EXIT trap replaces it with a one-minute shutdown when work ends.
sudo shutdown -c >/dev/null 2>&1 || true
sudo shutdown -h +360

run_cell() {
  local cell="$1"
  DISTILL_RUN_ID="$RUN_ID" DISTILL_SHUTDOWN=0 \
    bash infra/aws/phase-c-train.sh "$cell" >"$LOG_DIRECTORY/$cell.log" 2>&1
}

run_cell D3 &
d3_pid=$!
run_cell D10 &
d10_pid=$!

set +e
wait "$d3_pid"
d3_status=$?
wait "$d10_pid"
d10_status=$?
set -e

for cell in D3 D10; do
  aws s3 cp "$LOG_DIRECTORY/$cell.log" "$RUN_PREFIX/logs/phase-c-train-$cell.log"
  echo "===== $cell tail ====="
  tail -40 "$LOG_DIRECTORY/$cell.log"
done

if (( d3_status != 0 || d10_status != 0 )); then
  echo "Phase C data training failed: D3=$d3_status D10=$d10_status" >&2
  exit 1
fi

echo "Phase C data training completed: D3=$d3_status D10=$d10_status"
