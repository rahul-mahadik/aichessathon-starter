# Post-distillation architecture track

Phase D answers whether broader teacher supervision and additional capacity improve the evaluator.
It does not silently change search or inference: the original `search_engine.py` and reference
`QuantizedEvaluator` remain the frozen causal baseline.

The post-distillation track varies two independent deployment axes after the model artifacts exist:

| Axis | Baseline | Treatment |
|---|---|---|
| Search | current PVS/TT/qsearch | stronger PVS with aspiration, null move, LMR, futility, check extensions, countermoves, and delta-pruned qsearch |
| Runtime | rebuild sparse accumulators at every leaf | incrementally update quantized accumulators on move/unmove |

The stronger search is team-written and lives in `strong_search_engine.py`. It uses standard
published alpha-beta techniques and contains no third-party engine code or tuned parameter table.
The competition submission must never include the local Stockfish or Sunfish controls.

## Selecting an experimental engine

`weights/evaluator.json` may contain:

```json
{
  "antisymmetric": true,
  "search": "strong",
  "runtime": "incremental"
}
```

The defaults are `baseline` search and `reference` runtime, preserving all existing reports.
Benchmark packages create their candidate and opponent configuration independently, so one paired
match can isolate exactly one architecture change.

The incremental runtime rebuilds one perspective only when that perspective's king moves. Ordinary
moves, captures, promotion, en passant, castling-rook movement, null moves, and unwind use integer
feature-table deltas. Tests compare it to full recomputation at every push and pop.

An initial local 128/64/32 smoke profile used identical random weights and a 20,000-node strong
search. Both runtimes returned the same move with 11,346 evaluator calls. Reference inference took
1.987 seconds (5,709 calls/s); incremental inference took 1.507 seconds (7,530 calls/s), about 32%
more evaluator throughput. This is an implementation smoke result, not an Elo claim; Phase D's real
4.5 MB, 20 MB, and 40 MB models must be profiled separately on the competition-like CPU.

## Ready-to-run Phase D queue

GPU workers can wait for the validated dataset marker and begin immediately:

```bash
# Faster/larger GPU
DISTILL_RUN_ID=phase-d-20260903a \
  bash infra/aws/phase-d-train-queue.sh D100C40

# Second GPU, sequential cells so both models reuse its downloaded data
DISTILL_RUN_ID=phase-d-20260903a \
  bash infra/aws/phase-d-train-queue.sh D100C20 D100
```

Each queue waits for `dataset/phase-d-extra-90m/dataset.json`. It stops the GPU after its assigned
cells upload models and completion status.

On AWS DLAMIs, keep the dataset off the small RAM-backed `/tmp` filesystem and cache it across
sequential cells:

```bash
export DISTILL_WORK_ROOT=/home/ec2-user/aichessathon-work
export DISTILL_DATASET_CACHE_ROOT="$DISTILL_WORK_ROOT/dataset-cache/phase-d-20260903a"
```

After all three models exist, run the unconstrained evaluator test unchanged:

```bash
DISTILL_RUN_ID=phase-d-20260903a bash infra/aws/phase-d-evaluate.sh
```

Then run the search/runtime matrix, ideally one model per CPU worker:

```bash
DISTILL_RUN_ID=phase-d-20260903a \
  bash infra/aws/phase-d-search-matrix.sh phase-d-d100m-c4p5
DISTILL_RUN_ID=phase-d-20260903a \
  bash infra/aws/phase-d-search-matrix.sh phase-d-d100m-c20
DISTILL_RUN_ID=phase-d-20260903a \
  bash infra/aws/phase-d-search-matrix.sh phase-d-d100m-c40
```

The matrix includes paired 1k, 10k, and 100k fixed-node comparisons plus a wall-clock runtime cell.
Reports record model hashes, search mode, runtime mode, hardware, and Git revision.

## Decision order

1. Use the external evaluator holdout to reject models that did not learn the teacher better.
2. Use the frozen search at fixed nodes to measure the Phase D distillation effect.
3. Hold the selected evaluator fixed and compare baseline versus strong search.
4. Hold evaluator and strong search fixed and compare reference versus incremental inference under
   the clock.
5. Only then test a root/shallow policy head. MultiPV 8 is partial ranking supervision, so policy
   needs explicit legal-move negatives and its extra inference must beat ordinary move ordering.

Raw nodes, evaluator calls, and elapsed time should all be retained. Once pruning changes, equal
node counts no longer imply equal leaf-evaluation work.
