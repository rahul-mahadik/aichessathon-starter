# Frontier-inspired classical control

## Purpose

This control asks whether a conventional handcrafted evaluator plus a modern selective search can
beat the search-distillation approach under the same Chessathon runtime. It is deliberately
isolated in `controls/classical/`; the root submission and distilled engine are unchanged.

The control is submission-shaped and observes the strict current limits: Python 3.12, one CPU
core, 2 GB memory, no network or GPU, fixed preinstalled dependencies, and less than 50 MB
unzipped. It contains no third-party engine or model. The implementation is original, making it a
plausible entry rather than an impossible native-Stockfish straw man.

## Architecture review

The strongest CCRL engines are not useful drop-in controls: Stockfish, PlentyChess, Obsidian, and
other leaders combine highly tuned native search with NNUE evaluators. Shipping any of them is also
explicitly prohibited. Their source is used only to understand broadly published engine design.

The implemented design uses the overlap among these public architectures:

- [Stockfish](https://github.com/official-stockfish/Stockfish): iterative deepening, PVS,
  transposition bounds, selective pruning, and history-based ordering.
- [Berserk](https://github.com/jhonnold/berserk): a readable modern inventory of aspiration,
  null-move pruning, reverse futility pruning, LMR, and countermoves.
- [Ethereal 12.75](https://github.com/AndyGrant/Ethereal/tree/v12.75): the shape of a classical
  tapered evaluation, including pawn structure, mobility, rook files, king safety, and material
  phase.
- [Obsidian](https://github.com/gab8192/Obsidian): modern history and staged-ordering ideas.

No source, tables, tuned constants, or weights were copied. Piece-square values are generated from
simple board geometry, keeping the control independent and readable.

## Implemented engine

- tapered integer middlegame/endgame evaluation;
- geometric piece placement, pawn structure and passed pawns;
- mobility, bishop pair, rook-file activity, and king shelter;
- iterative deepening with aspiration windows and PVS;
- persistent, generation-aware transposition table with mate-distance normalization;
- quiescence with delta pruning and check evasions;
- null-move and reverse-futility pruning;
- late-move reductions and shallow quiet-move futility pruning;
- TT, MVV-LVA, killer, history, and countermove ordering;
- check extensions and exact fixed-node mode;
- the same time-allocation function as the experimental engine.

## Initial results

The optimized correctness-scale paired run used the public eight-position suite, both colours, two
rounds, and exactly 1,000 nodes per move:

| Candidate | Opponent | Games | W-D-L | Score | Estimated Elo | Failures |
|---|---|---:|---:|---:|---:|---:|
| Classical control | Existing handcrafted search | 32 | 16-12-4 | 68.8% | +136 | 0 |

This establishes a strong equal-search-quantity signal. The first 2 s + 0.05 s clock smoke test
exposed a hot-path problem: the richer initial evaluator was about four times slower than the
fallback and scored 28.1%. Precomputed geometric tables and bitboard masks reduced evaluation from
about 130 microseconds to 33 microseconds on the same local sample, approximately fallback speed.
Repeating the clock test then produced 10-3-3, a 71.9% score, with no failures. The full 120 s +
0.5 s tournament is still required; a short-clock smoke result is directional, not an Elo claim.

The 10,000-node local sweep was stopped after one long game because it was an inefficient use of
the workstation; it produced no aggregate result. The complete matrix should run on the AWS CPU
fleet after teacher labeling releases capacity.

The packaged control contains three Python files, measures 28,461 bytes unzipped, and contains no
native binaries or external weights.

## Required experiment matrix

1. Fixed nodes at 1k, 10k, and 100k against the frozen handcrafted engine.
2. Equal clock against that same engine under 120 s + 0.5 s.
3. Fixed nodes against the distilled small, C20, and C40 evaluators.
4. Equal clock against the strongest distilled candidate.
5. Record nodes/s, completed depth, failure rate, score, and confidence interval.

The classical engine becomes the alternative submission candidate only if it wins the equal-clock
test. Equal-node strength alone is research evidence, not a deployment decision.
