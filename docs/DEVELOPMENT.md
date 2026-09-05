# Development environment

## Local runtime

The root environment mirrors the competition's Python 3.12 package versions. The Python version is
pinned in `.python-version` so `uv` does not silently select a newer interpreter.

```bash
make setup
make gate
make play
```

If `uv` is not on `PATH`, set it explicitly:

```bash
make setup UV="$HOME/.local/bin/uv"
```

## Remotes and branches

Keep your GitHub fork as `origin` and the event repository as `upstream`:

```bash
git remote rename origin upstream
git remote add origin git@github.com:rahul-mahadik/aichessathon-starter.git
git fetch --all --prune
```

Use short-lived branches for engine experiments and preserve strong versions in directories outside
the candidate checkout so `benchmarks.run` can play them head-to-head.

## Benchmark loop

```bash
make gate
make bench BENCH_OPPONENT=../previous-agent BENCH_ROUNDS=10
make zip
unzip -l submission.zip
```

The benchmark suite swaps colours on every position and records JSON plus PGNs. A small run catches
crashes; hundreds of games are needed to support strength claims. Do final timing tests on Linux,
because macOS and the competition's single-core Linux container will not have identical performance.

## Training loop

Use the root CPU environment for data validation and smoke tests. Use the pinned CUDA container on a
Linux GPU host for actual training. Model export goes into `weights/`, which `make zip` includes by
default. TensorBoard logs, raw datasets, and checkpoints are ignored by Git.

AWS is the primary remote-compute path for this fork. See [`infra/aws/README.md`](../infra/aws/README.md)
for the EC2 benchmark worker, PyTorch 2.13 DLAMI training workflow, S3 artifact handling, and cost
guardrails.

Before an upload:

1. Confirm the model was trained by the team and document its data and code revision.
2. Load it with only the competition's approved packages.
3. Set all inference libraries to one CPU thread.
4. Verify import completes within 90 seconds.
5. Run paired games from several non-starting positions.
6. Inspect the ZIP root and its uncompressed size.

The local harness does not reproduce the production memory, network, read-only filesystem, or native
binary checks. The platform's validation log remains authoritative.
