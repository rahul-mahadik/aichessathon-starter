# Stockfish local research control

This directory wraps an unmodified Stockfish 18 process behind the same
`get_move(fen, time_left_ms)` interface as the Chessathon agents. It is deliberately a direct copy
of a frontier engine's decisions, not an independent implementation.

**Local research only. Never upload or package this directory.** Stockfish is a third-party engine
and the competition explicitly prohibits it. The wrapper is useful as:

- a search/evaluation ceiling;
- a fixed-node teacher-vs-student reference;
- a sanity check for openings and tactical positions;
- a control demonstrating how much strength remains in native move generation and mature search.

The process is constrained to one thread and 128 MB hash. Fixed-node mode uses the same
`AICHESSATHON_FIXED_NODES` environment variable as the team-written engines. Wall-clock mode uses
the same move-allocation function, minus 25 ms for UCI overhead.

Install Stockfish 18 locally or set `STOCKFISH_PATH`:

```bash
brew install stockfish
STOCKFISH_PATH=/path/to/stockfish .venv/bin/python -m benchmarks.run \
  --agent controls/stockfish_local --opponent controls/classical \
  --fixed-nodes 10000 --rounds 2 --workers 4 \
  --output benchmark-results/stockfish-vs-classical-10k.json
```

Stockfish is GPLv3. Its source and license are available from the
[official repository](https://github.com/official-stockfish/Stockfish) and the
[Stockfish 18 release](https://github.com/official-stockfish/Stockfish/releases/tag/sf_18).

## Initial results

Against `controls/classical` on the public eight-position paired suite:

| Stockfish budget | Games | W-D-L | Score | Estimated Elo | Failures |
|---|---:|---:|---:|---:|---:|
| 1,000 nodes/move | 16 | 15-1-0 | 96.9% | +597 | 0 |
| 10,000 nodes/move | 16 | 14-2-0 | 93.8% | +470 | 0 |

These are small, deterministic samples and the finite Elo estimates are unstable near 100%.
Their purpose is to establish an upper-bound control, not measure Stockfish's rating.

The initial local binary was Homebrew Stockfish 18 for Apple Silicon, SHA-256
`9fdf035807de93f26377eac8cb2a2304299d7514068f63dffcd9dfd1aac5f777`.
