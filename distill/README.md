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

On an AWS CPU worker, install the checksum-pinned official AVX2 build and measure the initial
100k-node/MultiPV-8 workload before scaling:

```bash
sudo bash infra/aws/install-stockfish.sh
TEACHER_LIMIT=8 bash infra/aws/teacher-pilot.sh
```

The pinned binary is Stockfish 18 from the project's official GitHub release. The installer exists
only on offline compute workers and nothing copies the binary into `weights/` or the submission.

## Reproducible Gigafish corpus

The sampler pins Gigafish to revision `47100399529ac17e9fdf2c8d0f49bfae89ae0c30`, samples
across four separated Parquet shards, validates and deduplicates FENs, and creates disjoint medium
and deep tiers. Its manifest records source URLs and SHA-256 hashes.

```bash
uv sync --group teacher
uv run --group teacher python -m distill.sample_gigafish \
  --output /tmp/aichessathon-teacher-corpus \
  --medium-positions 100000 \
  --deep-positions 10000 \
  --shards-per-tier 8
```

The first paid checkpoint is intentionally much smaller than the final target. It proves corpus,
label, training, and Elo quality before millions of examples are purchased.

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
