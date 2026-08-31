#!/usr/bin/env bash
set -euo pipefail

STOCKFISH_URL="${STOCKFISH_URL:-https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64-avx2.tar}"
STOCKFISH_SHA256="${STOCKFISH_SHA256:-536c0c2c0cf06450df0bfb5e876ef0d3119950703a8f143627f990c7b5417964}"
INSTALL_PATH="${STOCKFISH_INSTALL_PATH:-/usr/local/bin/stockfish}"
DOWNLOAD_DIR="$(mktemp -d)"
trap 'rm -rf "$DOWNLOAD_DIR"' EXIT

curl -fL "$STOCKFISH_URL" -o "$DOWNLOAD_DIR/stockfish.tar"
echo "$STOCKFISH_SHA256  $DOWNLOAD_DIR/stockfish.tar" | sha256sum --check
tar -xf "$DOWNLOAD_DIR/stockfish.tar" -C "$DOWNLOAD_DIR"
BINARY="$(find "$DOWNLOAD_DIR" -type f -name 'stockfish*' -perm -u+x | head -n 1)"
if [[ -z "$BINARY" ]]; then
  echo "No executable Stockfish binary found in the verified archive" >&2
  exit 1
fi
install -m 0755 "$BINARY" "$INSTALL_PATH"
UCI_OUTPUT="$(printf 'uci\nquit\n' | "$INSTALL_PATH")"
UCI_ID="$(grep -m1 '^id name Stockfish 18' <<<"$UCI_OUTPUT")"
if [[ -z "$UCI_ID" ]]; then
  echo "Installed binary did not identify itself as Stockfish 18" >&2
  exit 1
fi
printf '%s\n' "$UCI_ID"
