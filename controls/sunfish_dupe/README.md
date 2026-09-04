# Sunfish copied-engine research control

This directory contains a pinned source copy of the GPLv3
[Sunfish](https://github.com/thomasahle/sunfish) Python chess engine, adapted to the same
`get_move(fen, time_left_ms)` contract used by this repository. The copied revision is
`e396948acbd80cdcbad3dccfb07e878fba9df8f4` (Sunfish 2026).

**Local research only. Never upload this directory as a Chessathon submission.** It is a
third-party engine and exists solely to answer a different control question from the native
Stockfish binary: how strong is a compact, source-level copied engine when it runs through the
same Python process contract?

The copied file is `sunfish_upstream.py`; its copyright header and full GPLv3 license are
preserved. The only engine-source modification is a documented exact-node stop in `Searcher`,
used for symmetric fixed-node experiments. `agent.py` supplies FEN conversion, legal-move
validation, clock allocation, and game-history synchronization.

Run a local equal-node comparison with:

```bash
.venv/bin/python -m benchmarks.run \
  --agent controls/sunfish_dupe --opponent controls/classical \
  --fixed-nodes 10000 --rounds 2 --workers 4 \
  --output benchmark-results/sunfish-dupe-vs-classical-10k.json
```

This is not Stockfish and does not claim to reproduce Stockfish. Sunfish is a compact engine with
piece-square evaluation, iterative deepening MTD-bi search, transposition tables, quiescence,
null-move pruning, futility pruning, and late-move reductions. The real Stockfish wrapper remains
the frontier ceiling; this control measures a portable copied implementation.

## Initial results

Against `controls/classical` on the public eight-position paired suite:

| Sunfish budget | Games | W-D-L | Score | Estimated Elo | Failures |
|---|---:|---:|---:|---:|---:|
| 1,000 nodes/move | 16 | 10-2-4 | 68.8% | +137 | 0 |
| 10,000 nodes/move | 16 | 3-6-7 | 37.5% | -89 | 0 |

The reversal is not statistically decisive: the 95% Elo intervals are roughly -20 to +390 at
1k and -257 to +45 at 10k. It does establish that the control runs cleanly and that copying a
compact engine is not automatically a stronger baseline at larger node budgets. A larger paired
sample is required before interpreting the curve.
