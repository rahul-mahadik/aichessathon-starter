# Search-distilled engine architecture

The project tests whether expensive offline search can be amortized into a cheap evaluator without
pretending runtime search is unnecessary. Stockfish 18 is an offline teacher. Tournament moves are
produced only by our quantized evaluator and our Python search.

## Reproducible experiment boundary

The independent variable is teacher supervision:

1. handcrafted evaluation only;
2. shallow state-value pretraining;
3. fixed-node MultiPV child-value distillation;
4. deeper fixed-node MultiPV child-value distillation.

The search implementation, time manager, openings, and match harness remain fixed when comparing
these variants. Prediction metrics debug training; paired head-to-head Elo chooses the engine.

## Teacher data

`distill.annotate` emits a versioned raw JSONL record per root. Scores are always normalized to the
root side to move before storage. We retain cp/mate, WDL, candidate moves, requested budget, actual
nodes/depth/seldepth, and optionally PVs.

`distill.build_dataset` creates fixed groups:

```text
root s:       target V(s)
child T(s,a): target -Q(s,a)
candidate Q: retained in root perspective for ranking
```

Raw records are immutable. Different cp transforms, WDL targets, ranking margins, and loss weights
produce new processed datasets rather than relabeling with Stockfish.

## Student

Each perspective activates at most 32 features indexed by:

```text
(oriented king square, relative piece colour/type, oriented piece square)
```

The default network is:

```text
49,152 sparse features -> 128 accumulator (white and black)
concatenate side-to-move first -> 64 -> 32 -> tanh value
```

Only the sparse feature table is int8 in the first export format. This produces an approximately
5.3 MB model while keeping the dense layers simple. More aggressive integer inference is an
ablation, not a prerequisite.

## Runtime search

`search_engine.py` currently implements:

- iterative deepening with a hard wall-clock deadline;
- negamax alpha-beta with principal-variation search;
- a persistent bounded transposition table;
- quiescence search with check evasions;
- TT, MVV-LVA, promotion, killer, and history move ordering;
- repetition, fifty-move, insufficient-material, mate, and stalemate handling.

Timeout unwinding is guarded with `try/finally`, so an interrupted iteration cannot corrupt the
board. The last fully completed depth is always returned.

## Packaging boundary

The harness packages root `*.py` files plus `weights/`. It does not package `distill/`, `training/`,
datasets, Stockfish, checkpoints, or AWS scripts. `make zip` and the platform validator remain the
final authorities on the artifact.

## Next measured steps

1. Benchmark handcrafted search against minimax and freeze it as engine A.
2. Pretrain the same student on a tractable Gigafish sample.
3. Generate 100k-node MultiPV labels for a diverse pilot corpus.
4. Compare state-only and root-plus-child/ranking training.
5. Scale teacher nodes only after the pilot shows an Elo signal.

