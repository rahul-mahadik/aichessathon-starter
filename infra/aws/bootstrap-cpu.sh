#!/usr/bin/env bash
set -euo pipefail

UV_VERSION="${UV_VERSION:-0.11.7}"

sudo dnf install -y git jq unzip
curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
"$HOME/.local/bin/uv" --version

cat <<'EOF'
CPU worker ready.
Clone the team fork, check out the desired commit, then run:
  BENCH_WORKERS=4 BENCH_ROUNDS=25 bash infra/aws/benchmark.sh
EOF
