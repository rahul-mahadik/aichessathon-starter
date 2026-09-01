# AI Chessathon Experiment Report

Last updated: 2026-09-01 (America/Los_Angeles)  
Experiment run: `pilot-20260831a`  
Working branch: `codex/environment-setup`

## Executive summary

This report now separates three different questions: pure evaluator imitation on unseen positions,
fixed-node search quality, and wall-clock tournament strength. Earlier revisions conflated these
layers and moved too quickly from trace metrics to deployment conclusions.

The wall-clock result is clear: the initial distilled evaluators are not submission candidates. In
320 paired games each, the medium-only model scored 16.1% against the handcrafted fallback and the
combined medium-plus-deep model scored 15.5%. This does **not** establish that distillation failed;
it measures evaluator knowledge, inference cost, and search interaction together.

Trace diagnostics identified two causes:

1. The original full-size NN evaluator was about 1.8 times slower locally and completed one fewer
   search ply in the same time budget.
2. Although it ranked arbitrary candidate pairs better than the fallback, it selected Stockfish's
   top candidate less often.

A second controlled experiment therefore added an explicit top-move objective and trained a model
with a smaller accumulator. The new `fast-top` model is 1.18 MB, matches or slightly exceeds the
fallback's local evaluation throughput, reaches the same search depth, and selects Stockfish's top
candidate on 38.30% of a mixed training/validation trace shard versus 34.71% for the fallback.
However, it still scored only 15.2% against fallback in paired games. This disproves one-ply
top-candidate accuracy as a sufficient promotion metric for the current alpha-beta integration.

Turn-flip diagnostics then found a negamax integration violation: the learned evaluators did not
enforce antisymmetry when mover and opponent inputs swapped. On 5,000 positions, `fast-top` had a
mean `|v + v_flipped|` of 0.269 on a -1 to 1 scale, and 18.0% of pairs retained the same sign. An
optimized inference mode now reuses the sparse accumulator and antisymmetrizes only the dense head.
It retains search depth 4, scores 35.48% on the same mixed trace diagnostic, and is awaiting a
wall-clock paired game test as `fast-top-antisym`.

Current deployment leader: **handcrafted fallback**. Pure distillation and fixed-node search
leaders remain undetermined until the experiments below finish.

## Evaluation framework

| Test | Control | Question | Current state |
|---|---|---|---|
| Pure evaluator | No search or time limit; genuinely unseen deep labels | Did the student learn Stockfish values and move ordering? | Fresh holdout pending |
| Fixed-search | Same nodes per move; clocks disabled | Does the student improve the same search tree budget? | 1k/10k/100k harness ready |
| Tournament | Same wall clock and CPU | Does the complete engine win under competition constraints? | Initial results complete; diagnostics running |

## Data and teacher traces

The corpus was sampled from `lukesalamone/gigafish-3.8b-d10` at pinned revision
`47100399529ac17e9fdf2c8d0f49bfae89ae0c30`.

| Tier | Positions | Teacher budget | Shards |
|---|---:|---:|---:|
| Medium | 100,000 | 100,000 Stockfish nodes | 8 |
| Deep | 10,000 | 1,000,000 Stockfish nodes | 8 |
| Total | 110,000 | mixed | 16 |

Independent corpus validation:

- 110,000 records and 110,000 unique FENs; no duplicates.
- 827,767 legal candidate lines; 7.525 candidates per record on average.
- Complete MultiPV relative to each position's legal move count.
- WDL coverage: 100%.
- Centipawn coverage: 96.09%; remaining values include 32,383 mate scores.
- Observed Stockfish depths: 7 through 245.

Durable artifacts:

- Corpus: `s3://aichessathon-compute-artifactsbucket-snbc7mmwrkpq/artifacts/teacher/runs/pilot-20260831a/corpus/`
- Raw teacher traces: `s3://aichessathon-compute-artifactsbucket-snbc7mmwrkpq/artifacts/teacher/runs/pilot-20260831a/raw/`
- Reusable datasets: `s3://aichessathon-compute-artifactsbucket-snbc7mmwrkpq/artifacts/teacher/runs/pilot-20260831a/dataset/`
- Models: `s3://aichessathon-compute-artifactsbucket-snbc7mmwrkpq/artifacts/teacher/runs/pilot-20260831a/models/`
- Benchmarks: `s3://aichessathon-compute-artifactsbucket-snbc7mmwrkpq/artifacts/teacher/runs/pilot-20260831a/benchmarks/`

## Training and diagnostic metrics

| Model | Data | Architecture | Size | Validation ranking | Validation top move | Mixed-trace top move | Deployment status |
|---|---|---|---:|---:|---:|---:|---|
| `medium` | 100k medium | 128/64/32 | 4.63 MB | 59.42% | not recorded | not measured | Rejected for wall clock |
| `combined` | 100k medium + 10k deep | 128/64/32 | 4.56 MB | 60.39% | not recorded | 31.51% | Rejected for wall clock |
| `fast-top` | 100k medium | 32/32/16 | 1.18 MB | 60.43% | 31.82% | 38.30% | Rejected for wall clock |
| `fast-top-antisym` | existing `fast-top` weights | 32/32/16 | 1.18 MB | n/a | n/a | 35.48% | Tournament pending |
| `full-top` | 100k medium | 128/64/32 | 4.91 MB | 60.32% | 30.81% | 38.51% | Rejected for wall clock |

The 12,500-position diagnostic used retained medium-tier records containing 94,491 Stockfish
candidate positions. Because that shard mixes examples used for fitting with the deterministic
validation partition, it is **not** the pure unseen evaluator result. It only established that the
pipeline learned signal worth testing: `fast-top` pairwise ranking accuracy was 63.46%, compared
with 50.66% for the handcrafted evaluator.

## Pure evaluator test

A fresh, deduplicated position holdout with deep Stockfish labels will be stored separately from
all training data. For each model and the handcrafted fallback, the report will include root-value
MSE/MAE, candidate-value MSE/MAE, pairwise ranking accuracy, and top-move agreement. Runtime is not
a promotion criterion in this section.

Status: **pending fresh holdout generation and evaluation**.

Local throughput sample on 20,000 evaluations:

| Evaluator | Calls/second | Search result at identical budget |
|---|---:|---|
| Handcrafted | 19k-25k | depth 4 |
| Original combined | 13.8k | depth 3 |
| `fast-top` | 20.9k | depth 4 |

Absolute throughput varies by run and machine; the depth and relative throughput are the relevant
signals.

## Fixed-node search test

The benchmark harness now supports an exact per-move node cap, disables chess-clock deduction, and
uses a 24-hour per-move watchdog only for crash containment. Candidate and control use the same
search implementation and node ceiling.

| Candidate vs fallback | 1k nodes | 10k nodes | 100k nodes |
|---|---:|---:|---:|
| `medium` | pending | pending | pending |
| `combined` | pending | pending | pending |
| `fast-top` | pending | pending | pending |

The first sweep uses more games at cheap node budgets and fewer games at 100k, then expands any
promising cell. This is the experiment that tests whether search compute was amortized into the
weights.

## Wall-clock tournament test

Every completed comparison used 20 rounds over eight openings with both colors, for 320 games at a
5-second base clock, and reported zero failed terminations.

| Candidate | Opponent | W-D-L | Score | Estimated Elo | 95% score interval | Decision |
|---|---|---:|---:|---:|---:|---|
| `medium` | fallback | 42-19-259 | 16.09% | -287 | 12.29%-19.90% | Reject |
| `combined` | fallback | 35-29-256 | 15.47% | -295 | 11.86%-19.08% | Reject |
| `combined` | `medium` | 81-132-107 | 45.94% | -28 | 41.76%-50.12% | No wall-clock gain from deep labels |
| `fast-top` | fallback | 38-21-261 | 15.16% | -299 | 11.48%-18.83% | Reject |
| `full-top` | fallback | 16-27-277 | 9.22% | -397 | 6.47%-11.96% | Reject |
| `fast-top` | `full-top` | in progress | — | — | — | Speed/loss diagnostic |
| `fast-top-antisym` | fallback | pending | — | — | — | Negamax integration test |

The first three benchmark reports were generated from independently verified commit `3de6dfe`.
Their embedded Git field is `unknown` because SSM ran as root against an `ec2-user` checkout. This
provenance bug was fixed in `c72dd2a`; new reports record the safe Git revision, model names, and
model SHA-256 hashes.

## AWS execution and guardrails

- Region: `us-east-1`; profile: `aichessathon`.
- Account-level monthly AWS Budget: `$200` (`aichessathon-monthly-guardrail`). It is a visibility
  guardrail, not a hard resource stop.
- EC2 workers have a six-hour safety timer, encrypted disposable volumes, project/workload tags,
  and an instance role instead of static AWS credentials.
- Teacher generation used one `c7i.2xlarge` Spot worker.
- Training used short `g6.xlarge` and `g5.xlarge` on-demand sessions.
- Benchmarks use disposable `c7i.2xlarge` workers and terminate after reports are retained.
- No GPU is active. Two on-demand CPU workers are finishing wall-clock diagnostics.

The manual project ledger is currently below approximately `$2`, excluding negligible S3 request
and storage charges. This is an engineering estimate from instance lifetimes, not an AWS billing
statement. A final amount will be recorded after the active benchmarks terminate.

## Repository work completed

- AWS CloudFormation for private S3 artifacts, instance role, CPU/GPU launch templates, tagging,
  auto-stop, and the `$200` budget.
- Resumable, sharded Stockfish teacher workers with exact corpus manifests and SHA-256 hashes.
- Teacher trace schema, validation, dataset construction, sparse training, quantized export, and
  Numba runtime.
- Reusable S3 training datasets to avoid repeated raw conversion.
- Stable validation splitting and best-checkpoint export.
- Top-move-aware training objective and trace-level evaluator comparison.
- Paired, color-swapped benchmark reports with confidence intervals and Elo estimates.
- Benchmark provenance including Git revision and model hashes.

## Decision gate and next work

Do not decide whether distillation or the deeper labels worked from the wall-clock matches. First:

1. Generate and evaluate the genuinely unseen deep-labeled holdout.
2. Fill the 1k/10k/100k fixed-node matrix.
3. Compare the fixed-node curve with the wall-clock curve to separate evaluator quality from
   inference cost and search interaction.

Only then decide whether to scale teacher data, redesign the evaluator, or focus on deployment
speed. The handcrafted evaluator remains the submission default in the meantime.
