# AI Chessathon Experiment Report

Last updated: 2026-09-02 (America/Los_Angeles)
Experiment runs: `pilot-20260831a`, `scale-20260901a`
Working branch: `codex/environment-setup`

## Executive summary

This report now separates three different questions: pure evaluator imitation on unseen positions,
fixed-node search quality, and wall-clock tournament strength. Earlier revisions conflated these
layers and moved too quickly from trace metrics to deployment conclusions.

The wall-clock result is clear: the initial distilled evaluators are not submission candidates. In
320 paired games each, the medium-only model scored 16.1% against the handcrafted fallback and the
combined medium-plus-deep model scored 15.5%. This does **not** establish that distillation failed;
it measures evaluator knowledge, inference cost, and search interaction together.

Evaluator and search diagnostics identified three separate effects:

1. The original full-size NN evaluator was about 1.8 times slower locally and completed one fewer
   search ply in the same time budget.
2. On a fresh feature-disjoint holdout, every student ranked arbitrary candidate pairs better than
   the fallback, but every student selected Stockfish's exact top candidate less often.
3. The current students also lose when both engines receive exactly the same number of search
   nodes, so inference speed is not the only problem.

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
It retains search depth 4 and improved the wall-clock score from 15.2% to 22.2%, but it still loses
decisively to the fallback.

Current deployment and fixed-node leader: **handcrafted fallback**. The pure evaluator result is
mixed but positive: `combined` is the best student on value error, `fast-top` is the best on broad
pairwise ordering, and the fallback remains best on exact top-move agreement. The weights therefore
contain real Stockfish signal, but not yet in a form that improves alpha-beta move choice.

A train-time correctness pass added antisymmetry loss and four-times ranking weight for pairs
involving the teacher's top three moves. It worked mechanically but did not improve the 110k-scale
fixed-node result. Phase B has now completed label generation, disagreement mining, clean holdout
selection, and dataset construction for one deliberately bounded 10x experiment. Four controlled
students are training; evaluator and fixed-node conclusions remain conditional on their results.

## Evaluation framework

| Test | Control | Question | Current state |
|---|---|---|---|
| Pure evaluator | No search or time limit; genuinely unseen deep labels | Did the student learn Stockfish values and move ordering? | Complete on 2,000 positions |
| Fixed-search | Same nodes per move; clocks disabled | Does the student improve the same search tree budget? | First 1k/10k/100k sweep complete |
| Tournament | Same wall clock and CPU | Does the complete engine win under competition constraints? | Complete for current variants |

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
- Unseen evaluator holdout and results: `s3://aichessathon-compute-artifactsbucket-snbc7mmwrkpq/artifacts/teacher/runs/eval-20260901a/`
- Scale corpus and traces: `s3://aichessathon-compute-artifactsbucket-snbc7mmwrkpq/artifacts/teacher/runs/scale-20260901a/`

## Phase B progress recap

Phase B tests the data-scale and information-selection hypotheses left open by the 110,000-position
pilot. Its target dataset is 1,000,000 medium positions plus 100,000 deep positions selected because
additional Stockfish search materially changes their value or move ordering. This is intentionally
different from merely adding more randomly sampled deep labels.

### Corpus and label generation: complete

The disjoint corpus excludes the original 110,000 training positions and the 2,004 evaluator
holdout/placeholder positions at the model-visible feature level. It contains 1,000,000 medium
positions and a separate 200,000-position depth-mining pool. All tiers use 256 shards. The mining
pool was labeled at three node budgets on the same positions so changes with search depth can be
measured directly.

| Label tier | Unique positions | Nodes per position | Output shards | Status |
|---|---:|---:|---:|---|
| Medium | 1,000,000 | 100,000 | 256/256 | Complete |
| Deep mining: shallow | 200,000 | 10,000 | 256/256 | Complete |
| Deep mining: medium | 200,000 | 100,000 | 256/256 | Complete |
| Deep mining: deep | 200,000 | 1,000,000 | 256/256 | Complete |

This produced 1,600,000 teacher records over 1,200,000 unique positions, representing a nominal
322 billion requested Stockfish nodes. The four compressed raw tiers occupy approximately 208 MB
in S3. All 32 worker/tier completion markers report `failed=0`. The final 256 deep-1M gzip shards
passed an end-to-end integrity check and contain exactly 200,000 records.

Artifacts are retained under:

`s3://aichessathon-compute-artifactsbucket-snbc7mmwrkpq/artifacts/teacher/runs/scale-20260901a/`

### Compute execution

Labeling launched on eight `c7i.8xlarge` instances in `us-east-1`: 32 vCPUs and 64 GiB per worker,
or 256 vCPUs at peak. Each shard used one single-threaded Stockfish process, so the workload was
CPU-bound and GPUs would not have helped. The first workers launched around 08:00 UTC on 2026-09-02;
the final S3 object arrived at 09:25 UTC. Workers terminated after completing their assignments,
and no EC2 instances remain active.

Seven workers finished first, leaving one worker responsible for the last 32 deep-1M shards. The
fixed modulo assignment and whole-shard uploads made those in-flight shards impossible to reassign
without restarting them. A duplicate retry briefly oversubscribed the last 32-vCPU worker with two
copies of the same work. One command completed and shut the instance down; the other became
undeliverable. The resulting S3 tier passed gzip and record-count validation.

Operational fixes and lessons from the run:

- SSM cannot be assumed to provide `HOME`; worker setup now uses an explicit repository default.
- Environment setup is reusable, so a retry does not fail because its virtual environment exists.
- Future scale runs should use 4-8 times more, smaller shards, dynamic work claiming, per-shard
  leases, and periodic checkpoints. This removes fixed-worker stragglers and prevents duplicate
  writers.
- Completed shards are checked in S3 before work begins, which makes clean retries resumable.

### Controlled ablation preparation: complete

Label generation alone does not answer whether Phase B improved the distilled evaluator. The
remaining experiment holds architecture, initialization, loss weights, and optimizer steps fixed
across four data ablations:

| Model | Base data | Extra positions | Extra labels | Question |
|---|---|---|---|---|
| `phase-b-m` | 1M medium | none | none | Scale baseline |
| `phase-b-r-deep` | 1M medium | 100k random mining-pool positions | 1M nodes | Does random deep data help? |
| `phase-b-h-medium` | 1M medium | 100k selected hard positions | 100k nodes | Does hard-position selection help? |
| `phase-b-h-deep` | 1M medium | the same 100k hard positions | 1M nodes | Does deeper search add transferable information? |

The primary causal comparisons are `phase-b-h-deep` versus `phase-b-h-medium` for label depth and
`phase-b-h-deep` versus `phase-b-r-deep` for information-directed selection. A deterministic
20,000-position subset of the mining pool is reserved before selection as a common evaluation
holdout; none of the four models trains on it.

Disagreement uses expected WDL change, candidate ranking reversals, top-three displacement, and a
large explicit weight for best-move flips. Selection is stratified across best-move flips,
top-three reorderings, outcome changes, quiet disagreements, and random coverage rather than taking
the largest raw centipawn deltas. The 10k/100k/1M trajectory also defines easy, medium, deep,
unstable, and transitional holdout buckets.

Mining completed over all 200,000 positions. After reserving the 20,000-position holdout, the
100,000 selected-hard set contains exactly 35,000 top-move flips, 20,000 top-three reorderings,
20,000 outcome changes, 15,000 quiet disagreements, and 10,000 random-coverage examples. The
100,000-position random control was sampled from the same 180,000-position training pool. Its
55,605-position overlap with the hard set is the expected overlap for two samples of that size,
not a train/evaluation leak.

| Search-depth transition | Top move changed | Mean root-value delta | Mean ranking disagreement |
|---|---:|---:|---:|
| 10k to 100k nodes | 41.57% | 0.0552 | 0.0653 |
| 100k to 1M nodes | 31.94% | 0.0312 | 0.0360 |

The clean holdout contains 6,753 easy, 1,409 medium, 1,200 deep, 2,472 unstable, and 8,166
transitional examples. Selection hashes, exact input lists, and all bucket traces are retained in
the run's `mined/phase-b-ablation/` prefix. Dataset conversion produced one 1,000,000-record base
component and three 100,000-record treatment components; the four training recipes compose these
without duplicating the base data in S3.

### Training and evaluation: in progress

All four models use the corrected 128/64/32 architecture, seed 7, identical antisymmetry and
top-move-aware losses, and exactly 500 optimizer steps in each of 20 epochs. The four jobs run
concurrently on one `g5.xlarge`: one process per vCPU, approximately 6.2 of 23 GiB device memory,
and 99% aggregate A10G utilization. A prepared `c7i.8xlarge` will run the unconstrained
pure-evaluator tests followed by equal-node 1k/10k/100k search tests. Equal-wall-clock tournaments
remain gated on success in the first two experiments.

Phase B data generation and preparation are complete. Distillation and evaluation are still in
progress, so the run does not yet establish a chess-strength result.

## Training and diagnostic metrics

| Model | Data | Architecture | Size | Validation ranking | Validation top move | Mixed-trace top move | Deployment status |
|---|---|---|---:|---:|---:|---:|---|
| `medium` | 100k medium | 128/64/32 | 4.63 MB | 59.42% | not recorded | not measured | Rejected for wall clock |
| `combined` | 100k medium + 10k deep | 128/64/32 | 4.56 MB | 60.39% | not recorded | 31.51% | Rejected for wall clock |
| `fast-top` | 100k medium | 32/32/16 | 1.18 MB | 60.43% | 31.82% | 38.30% | Rejected for wall clock |
| `fast-top-antisym` | existing `fast-top` weights | 32/32/16 | 1.18 MB | n/a | n/a | 35.48% | Better than direct mode, still rejected |
| `full-top` | 100k medium | 128/64/32 | 4.91 MB | 60.32% | 30.81% | 38.51% | Rejected for wall clock |
| `corrected-64` | 100k medium + 10k deep | 64/48/24 | 2.58 MB | 61.49% | 32.51% | not measured | No fixed-node gain |
| `corrected-128` | 100k medium + 10k deep | 128/64/32 | 5.12 MB | 60.99% | 31.98% | not measured | Matches prior fixed-node best |

The 12,500-position diagnostic used retained medium-tier records containing 94,491 Stockfish
candidate positions. Because that shard mixes examples used for fitting with the deterministic
validation partition, it is **not** the pure unseen evaluator result. It only established that the
pipeline learned signal worth testing: `fast-top` pairwise ranking accuracy was 63.46%, compared
with 50.66% for the handcrafted evaluator.

## Pure evaluator test

A fresh holdout was sampled deterministically from the pinned Gigafish revision after excluding all
110,000 training positions at the evaluator's feature level: piece placement plus side to move.
This is stricter than exact-FEN exclusion because positions that differ only in counters or other
state invisible to the model are also removed. The holdout contains 2,000 unique positions and
15,048 legal candidate lines, all labeled by Stockfish at 1,000,000 nodes with MultiPV 8. Runtime
is not a criterion in this experiment.

| Evaluator | Root MSE | Root MAE | Candidate MSE | Candidate MAE | Pairwise accuracy | Top-move agreement |
|---|---:|---:|---:|---:|---:|---:|
| Handcrafted fallback | 0.3856 | 0.4969 | 0.3803 | 0.4954 | 51.89% | **34.00%** |
| `medium` | 0.3922 | 0.4506 | 0.3834 | 0.4419 | 61.17% | 29.55% |
| `combined` | **0.3804** | **0.4382** | **0.3747** | **0.4308** | 60.66% | 28.05% |
| `fast-top` | 0.3831 | 0.4551 | 0.3851 | 0.4459 | **62.00%** | 29.50% |
| `full-top` | 0.3988 | 0.4734 | 0.3956 | 0.4678 | 61.52% | 30.55% |
| `fast-top-antisym` | 0.3877 | 0.4664 | 0.3830 | 0.4608 | 60.86% | 28.85% |
| `corrected-64` | 0.4387 | 0.5233 | 0.4307 | 0.5165 | 61.91% | 30.85% |
| `corrected-128` | 0.4503 | 0.5389 | 0.4465 | 0.5348 | 61.34% | 31.20% |

This establishes **partial distillation success** independent of runtime. All students improve
pairwise ordering substantially, and the initial models reduce absolute value error; `combined`
also improves both root and candidate MSE. The corrected pilot regresses on value error while only
slightly improving top-move agreement over `fast-top`. Every student remains below the fallback's
34.0% exact top-move agreement. The students have learned the teacher's broad value landscape, but
the loss and representation do not concentrate that knowledge strongly enough on the best move.

### Runtime diagnostic (not part of the pure evaluator test)

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

| Candidate vs fallback | 1k nodes (320 games) | 10k nodes (80 games) | 100k nodes (16 games) |
|---|---:|---:|---:|
| `medium` | 12.50% / -338 Elo | 15.63% / -293 Elo | 15.63% / -293 Elo |
| `combined` | 15.63% / -293 Elo | 3.13% / -597 Elo | 9.38% / -394 Elo |
| `fast-top` | 18.75% / -255 Elo | 0.00% / unbounded | 12.50% / -338 Elo |
| `corrected-64` | 12.50% / -338 Elo | not run | not run |
| `corrected-128` | 18.75% / -255 Elo | not run | not run |

The first sweep uses more games at cheap node budgets and fewer games at 100k, then expands any
promising cell. All nine cells disabled clock enforcement, used the same node ceiling on both
sides, and had zero failed terminations. The 100k cells are directional because they contain only
16 games. No current student shows evidence that search compute was successfully amortized into
the weights; the degradation at 10k also points to a depth/search interaction that deserves direct
analysis.

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
| `fast-top` | `full-top` | 129-114-77 | 58.13% | +57 | 53.81%-62.44% | Smaller model wins |
| `fast-top-antisym` | fallback | 35-72-213 | 22.19% | -218 | 18.44%-25.93% | Better, still reject |

The first three benchmark reports were generated from independently verified commit `3de6dfe`.
Their embedded Git field is `unknown` because SSM ran as root against an `ec2-user` checkout. This
provenance bug was fixed in `c72dd2a`; new reports record the safe Git revision, model names, and
model SHA-256 hashes.

## AWS execution and guardrails

- Region: `us-east-1`; profile: `aichessathon`.
- Project-tagged monthly operating target: `$200` (`aichessathon-monthly-target`).
- Project-tagged emergency budget: `$400` (`aichessathon-monthly-emergency-stop`) with an automatic
  SSM action targeting active project EC2 instances.
- An atomic DynamoDB reservation gate charges every launch its worst-case six-hour cost before EC2
  starts and refuses launches above `$400`. AWS billing data is delayed, so this reservation gate,
  not AWS Budgets alone, is the immediate control.
- EC2 workers have a six-hour safety timer, encrypted disposable volumes, project/workload tags,
  and an instance role instead of static AWS credentials.
- Pilot teacher generation used one `c7i.2xlarge` Spot worker.
- Phase B label generation used eight `c7i.8xlarge` workers: 256 vCPUs total, matching the regional
  quota.
- Training used short `g6.xlarge` and `g5.xlarge` on-demand sessions.
- Benchmarks use disposable `c7i.2xlarge` workers and terminate after reports are retained.
- All eight Phase B teacher workers are terminated. One `g5.xlarge` training worker and one reused
  `c7i.8xlarge` evaluation worker are active for the controlled ablation.
- Two untagged stopped instances created in April, each with an attached 16 GB `gp3` volume, remain
  in the account. They predate this project and were not modified.

The reservation ledger currently authorizes `$185.00` of worst-case monthly project spend,
including a conservative `$20` baseline for prior runs. This is intentionally higher than expected
actual cost because reservations are not released when workers finish early.

**Historical account warning:** before project-tag filtering was activated, the account-wide budget
reported a `$265.31` forecast. Cost Explorer reports approximately `$252.74` account-wide for
August, including `$243.96` of EC2 compute, `$5.08` of EC2 Other, and `$3.70` of VPC charges. The
new budgets filter on the now-active `Project` cost-allocation tag; the older account totals remain
separate from the project reservation ledger.

## Repository work completed

- AWS CloudFormation for private S3 artifacts, instance role, CPU/GPU launch templates, tagging,
  auto-stop, project-tagged `$200`/`$400` budgets, an automatic EC2 stop action, and an atomic
  worst-case launch ledger.
- Resumable, sharded Stockfish teacher workers with exact corpus manifests and SHA-256 hashes.
- Teacher trace schema, validation, dataset construction, sparse training, quantized export, and
  Numba runtime.
- Reusable S3 training datasets to avoid repeated raw conversion.
- Stable validation splitting and best-checkpoint export.
- Top-move-aware training objective and trace-level evaluator comparison.
- Paired, color-swapped benchmark reports with confidence intervals and Elo estimates.
- Clock-free, exact-node benchmark mode and a complete first 1k/10k/100k sweep.
- Feature-level holdout exclusion and a separate 2,000-position, deep-labeled evaluator corpus.
- Train-time antisymmetry loss, top-k-weighted ranking, and sparse-accumulator reuse.
- Paired-depth disagreement mining for identical 10k/100k/1M teacher positions.
- Benchmark provenance including Git revision and model hashes.

## Decision gate and next work

The three experiments now support a narrower diagnosis:

1. Distillation learned broad Stockfish value and ordering signal on unseen positions.
2. That signal does not improve the current alpha-beta engine at an equal node count.
3. Inference speed compounds the problem for the larger networks under a wall clock, but it is not
   the primary explanation because the students already lose the fixed-node test.

Train-time antisymmetry and top-k weighting did not improve the 110k-scale fixed-node cell, so those
changes alone are insufficient. Phase B's four controlled students are now training. Proceed to
10M/1M or an explicit policy head only if a resulting student improves unseen top-k agreement and
at least one fixed-node cell. The handcrafted evaluator remains the submission default in the
meantime.
