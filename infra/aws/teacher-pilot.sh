#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${UV_BIN:-}" ]]; then
  UV_BIN="$(command -v uv || true)"
  UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
fi
STOCKFISH_BIN="${STOCKFISH_PATH:-/usr/local/bin/stockfish}"
INPUT_PATH="${TEACHER_INPUT:-benchmarks/openings.epd}"
OUTPUT_PATH="${TEACHER_OUTPUT:-/tmp/aichessathon-teacher-pilot.jsonl.gz}"
NODE_BUDGET="${TEACHER_NODES:-100000}"
MULTIPV="${TEACHER_MULTIPV:-8}"
POSITION_LIMIT="${TEACHER_LIMIT:-8}"

if [[ ! -x "$STOCKFISH_BIN" ]]; then
  echo "Stockfish 18 not found at $STOCKFISH_BIN" >&2
  echo "Run: sudo bash infra/aws/install-stockfish.sh" >&2
  exit 1
fi

"$UV_BIN" sync --frozen --python 3.12
STARTED="$(date +%s)"
"$UV_BIN" run python -m distill.annotate \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH" \
  --stockfish "$STOCKFISH_BIN" \
  --nodes "$NODE_BUDGET" \
  --multipv "$MULTIPV" \
  --limit "$POSITION_LIMIT"
FINISHED="$(date +%s)"
ELAPSED="$((FINISHED - STARTED))"

echo "teacher pilot: positions=$POSITION_LIMIT nodes=$NODE_BUDGET multipv=$MULTIPV seconds=$ELAPSED"
echo "output: $OUTPUT_PATH"
if [[ -n "${AICHESSATHON_ARTIFACTS_URI:-}" ]]; then
  aws s3 cp "$OUTPUT_PATH" \
    "${AICHESSATHON_ARTIFACTS_URI%/}/teacher/pilots/$(basename "$OUTPUT_PATH")"
fi
