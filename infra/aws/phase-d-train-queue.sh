#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
  echo "at least one cell is required: D100, D100C20, or D100C40" >&2
  exit 2
fi

shutdown_host() {
  if [[ "${DISTILL_SHUTDOWN:-1}" == "1" ]]; then
    sudo shutdown -h +1
  fi
}
trap shutdown_host EXIT

for cell in "$@"; do
  DISTILL_SHUTDOWN=0 bash infra/aws/phase-d-train-when-ready.sh "$cell"
done
