# Offline distillation pipeline

Everything in this directory is training infrastructure. It is never included in
`submission.zip`; Stockfish is an offline label generator only.

## Local vertical slice

Install Stockfish 18 and run the two-position end-to-end smoke test:

```bash
brew install stockfish
make distill-e2e-smoke STOCKFISH=/opt/homebrew/bin/stockfish
```

The flow is:

```text
FEN/EPD
  -> fixed-node Stockfish MultiPV
  -> raw versioned JSONL.gz
  -> sparse root + child-state NPZ groups
  -> value + pairwise-ranking training
  -> quantized weights/nnue.npz
```

`annotate.py` configures Stockfish with one thread and a fixed node or depth budget. Each JSONL
record preserves raw centipawns or mates, WDL when available, requested budget, actual depth,
seldepth, nodes, candidate moves, and optional PVs. Target transformations happen later, so they
can be changed without rerunning the expensive teacher.

## Create a teacher shard

```bash
uv run python -m distill.annotate \
  --input positions.epd \
  --output training/data/teacher/part-00000.jsonl.gz \
  --stockfish /path/to/stockfish \
  --nodes 100000 \
  --multipv 8
```

Use `--keep-pv` only when the lines are needed; they increase storage substantially. Use a unique
output object for every teacher version, node budget, sampling source, and code revision.

## Build sparse training shards

```bash
uv run python -m distill.build_dataset \
  training/data/teacher/part-*.jsonl.gz \
  --output training/data/distilled/100k-nodes \
  --records-per-shard 100000
```

Every group holds a root target and up to eight sign-flipped candidate child targets. Candidate
scores remain in root perspective for the pairwise ranking loss.

## Train and export

```bash
uv run python -m training.train_distilled \
  --data training/data/distilled/100k-nodes \
  --device cuda \
  --epochs 10 \
  --output weights/nnue.npz
```

The export contains an int8 king-conditioned feature table and small float32 dense layers. The
runtime loads it with `allow_pickle=False`; no training code or third-party engine is shipped.

