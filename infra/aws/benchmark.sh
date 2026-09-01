#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${UV_BIN:-}" ]]; then
  UV_BIN="$(command -v uv || true)"
  UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
fi
WORKERS="${BENCH_WORKERS:-1}"
ROUNDS="${BENCH_ROUNDS:-10}"
BASE_MS="${BENCH_BASE_MS:-10000}"
FIXED_NODES="${BENCH_FIXED_NODES:-}"
AGENT="${BENCH_AGENT:-.}"
OPPONENT="${BENCH_OPPONENT:-baselines/minimax}"
OUTPUT="${BENCH_OUTPUT:-benchmark-results/aws-$(date -u +%Y%m%dT%H%M%SZ).json}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Capture the exact EC2 hardware image in the result when IMDSv2 is available.
TOKEN="$(curl --silent --fail --max-time 1 \
  --request PUT --header 'X-aws-ec2-metadata-token-ttl-seconds: 300' \
  http://169.254.169.254/latest/api/token || true)"
if [[ -n "$TOKEN" ]]; then
  export AWS_EC2_INSTANCE_ID="$(curl --silent --fail --max-time 1 \
    --header "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id)"
  export AWS_EC2_INSTANCE_TYPE="$(curl --silent --fail --max-time 1 \
    --header "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-type)"
  export AWS_EC2_AMI_ID="$(curl --silent --fail --max-time 1 \
    --header "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/ami-id)"
fi

"$UV_BIN" sync --frozen --python 3.12
EXTRA_ARGUMENTS=()
if [[ -n "$FIXED_NODES" ]]; then
  EXTRA_ARGUMENTS+=(--fixed-nodes "$FIXED_NODES")
fi

"$UV_BIN" run python -m benchmarks.run \
  --agent "$AGENT" \
  --opponent "$OPPONENT" \
  --rounds "$ROUNDS" \
  --base-ms "$BASE_MS" \
  --workers "$WORKERS" \
  --output "$OUTPUT" \
  "${EXTRA_ARGUMENTS[@]}"

if [[ -n "${AICHESSATHON_ARTIFACTS_URI:-}" ]]; then
  aws s3 cp "$OUTPUT" "${AICHESSATHON_ARTIFACTS_URI%/}/benchmarks/$(basename "$OUTPUT")"
fi
