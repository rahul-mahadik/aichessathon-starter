# Classical control engine

This is the non-distillation control. It is a complete, submission-shaped Python agent using an
original handcrafted evaluation and a modern alpha-beta search. It intentionally shares neither
runtime code nor weights with the search-distilled agent.

The architecture takes standard ideas used broadly by Stockfish, Ethereal, Berserk, and other
classical engines: tapered evaluation, iterative deepening, aspiration windows, PVS, a persistent
transposition table, quiescence, null-move pruning, late-move reductions, futility pruning,
killer/history/countermove ordering, and check extensions. The implementation is original Python;
it does not contain another engine or its weights.

Run paired fixed-node and competition-clock comparisons from the repository root:

```bash
.venv/bin/python -m benchmarks.run --agent controls/classical --opponent . \
  --fixed-nodes 10000 --rounds 5 --workers 4 \
  --output benchmark-results/classical-vs-distilled-10k.json

.venv/bin/python -m benchmarks.run --agent controls/classical --opponent . \
  --base-ms 120000 --increment-ms 500 --rounds 5 --workers 4 \
  --output benchmark-results/classical-vs-distilled-clock.json
```

To prove the directory has the same basic package shape as a submission, run the package builder
with this directory as the working directory. Do not replace the actual submission from here until
the benchmark results justify doing so.
