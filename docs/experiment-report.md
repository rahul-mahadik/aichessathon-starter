# AI Chessathon Experiment Report

Last updated: 2026-09-02 (America/Los_Angeles)
Experiment runs: `pilot-20260831a`, `scale-20260901a`, `phase-c-20260902a`
Working branch: `codex/environment-setup`

## Executive summary

This report now separates three different questions: pure evaluator imitation on unseen positions,
fixed-node search quality, and wall-clock tournament strength. Earlier revisions conflated these
layers and moved too quickly from trace metrics to deployment conclusions.

The initial 110,000-position students were not submission candidates. Their best wall-clock score
was 22.19% against the handcrafted fallback, and they also lost when both engines received the same
node budget. That result mixed evaluator knowledge, inference cost, and search interaction, so it
did not by itself disprove distillation.

Phase B changes the research conclusion. On a genuinely unseen 2,000-position, 1M-node teacher
set, all four scaled students beat the fallback on value error and pairwise ordering. The
medium-only `phase-b-m` also reaches 34.5% exact top-move agreement versus 34.0% for the fallback.
At exactly 1,000 search nodes per move it scores 65.63% over 320 paired games, approximately +112
Elo with a +84 to +142 Elo interval. This is clear evidence that the distilled evaluator can make a
fixed shallow search tree stronger.

The controlled data ablation rejects the current deep-mining recipe. At 1k nodes, random-deep,
selected-medium, and selected-deep additions score 56.25%, 50.00%, and 43.75% respectively. Their
10k and directional 100k results show the same ordering. Deeper labels slightly improve broad
one-ply ordering but reduce exact top-move agreement and search strength; medium-only breadth is
the useful signal at this scale.

Deployment remains a separate negative result. `phase-b-m` improves the wall-clock score to 42.34%
(about -54 Elo), far above the prior students but still decisively below the fallback. The 32/32/16
and 64/48/24 compressions score 36.72% and 25.47% respectively under the clock; smaller is not
monotonically stronger. Current deployment leader: **handcrafted fallback**. Current fixed-node
leader at 1k: **`phase-b-m`**.

Phase C therefore treats research strength and deployment efficiency as separate tracks. The
research track is now scaling the successful broad 100k-node label recipe across nested 1M, 3M,
and 10M datasets, while independently scaling student capacity from 4.5 MB toward 10, 20, and
40 MB. These first cells retain the Phase B objective, seed, batch size, and 20-epoch recipe. The
data axis runs 20 complete passes, so optimizer compute scales with corpus size. Wall-clock tests
remain a periodic deployability check, not a gate on this scaling study.

## Evaluation framework

| Test | Control | Question | Current state |
|---|---|---|---|
| Pure evaluator | No search or time limit; genuinely unseen deep labels | Did the student learn Stockfish values and move ordering? | Complete on 2,000 positions |
| Fixed-search | Same nodes per move; clocks disabled | Does the student improve the same search tree budget? | Phase B sweep complete; Phase C scaling active |
| Tournament | Same wall clock and CPU | Does the complete engine win under competition constraints? | Complete for Phase B candidates |

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
- Phase C scaling artifacts: `s3://aichessathon-compute-artifactsbucket-snbc7mmwrkpq/artifacts/teacher/runs/phase-c-20260902a/`

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

### Controlled training and evaluation: complete

All four models use the corrected 128/64/32 architecture, seed 7, identical antisymmetry and
top-move-aware losses, and exactly 500 optimizer steps in each of 20 epochs. The four jobs completed
concurrently on one `g5.xlarge`: one process per vCPU, approximately 6.2 of 23 GiB device memory,
and 99% aggregate A10G utilization. Four `c7i.8xlarge` workers then ran the unconstrained evaluator
tests and equal-node 1k/10k/100k search tests in parallel. The four-model controlled experiment is
complete. Post-ablation follow-ups expanded the full model's 10k cell and trained 32/32/16 and
64/48/24 medium-only capacity variants. Those architecture tests are reported separately and do
not alter the controlled data-ablation conclusion.

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
| `phase-b-m` | 1M medium | 128/64/32 | 4.47 MB | 63.42% | 34.71% | not measured | Phase B scale leader |
| `phase-b-r-deep` | 1M medium + 100k random deep | 128/64/32 | 4.65 MB | 63.48% | 34.77% | not measured | Phase B control |
| `phase-b-h-medium` | 1M medium + 100k selected medium | 128/64/32 | 4.48 MB | 63.35% | 34.30% | not measured | Phase B selection treatment |
| `phase-b-h-deep` | 1M medium + 100k selected deep | 128/64/32 | 4.39 MB | 63.37% | 34.31% | not measured | Phase B depth treatment |
| `phase-b-m-fast` | 1M medium | 32/32/16 | 1.17 MB | 62.69% | 34.50% | not measured | Fails 1k fixed-node gate |
| `phase-b-m-mid` | 1M medium | 64/48/24 | 2.11 MB | 62.99% | 34.47% | not measured | Passes 1k fixed-node gate |

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
| Handcrafted fallback | 0.3856 | 0.4969 | 0.3803 | 0.4954 | 51.89% | 34.00% |
| `medium` | 0.3922 | 0.4506 | 0.3834 | 0.4419 | 61.17% | 29.55% |
| `combined` | 0.3804 | 0.4382 | 0.3747 | 0.4308 | 60.66% | 28.05% |
| `fast-top` | 0.3831 | 0.4551 | 0.3851 | 0.4459 | 62.00% | 29.50% |
| `full-top` | 0.3988 | 0.4734 | 0.3956 | 0.4678 | 61.52% | 30.55% |
| `fast-top-antisym` | 0.3877 | 0.4664 | 0.3830 | 0.4608 | 60.86% | 28.85% |
| `corrected-64` | 0.4387 | 0.5233 | 0.4307 | 0.5165 | 61.91% | 30.85% |
| `corrected-128` | 0.4503 | 0.5389 | 0.4465 | 0.5348 | 61.34% | 31.20% |
| `phase-b-m` | 0.3354 | 0.4457 | 0.3328 | 0.4428 | 63.74% | **34.50%** |
| `phase-b-r-deep` | 0.3322 | 0.4418 | 0.3238 | 0.4344 | 64.29% | 33.05% |
| `phase-b-h-medium` | **0.3253** | **0.4368** | **0.3187** | **0.4316** | 64.22% | 32.65% |
| `phase-b-h-deep` | 0.3283 | 0.4391 | 0.3229 | 0.4344 | **64.75%** | 32.40% |
| `phase-b-m-fast` | 0.3360 | 0.4478 | 0.3265 | 0.4422 | 63.02% | 33.05% |
| `phase-b-m-mid` | 0.3410 | 0.4485 | 0.3353 | 0.4450 | 63.31% | 33.15% |

This establishes **distillation success** independent of runtime. All students improve pairwise
ordering substantially, and the Phase B students beat the fallback on root and candidate value
error. `phase-b-h-medium` has the best value imitation and `phase-b-h-deep` the best broad ordering,
but both lose exact top-move agreement. `phase-b-m` is the first student to exceed the fallback's
34.0% top-move agreement, reaching 34.5%. The ablation shows that deeper or
information-selected labels do not automatically improve the decision-relevant metric.

The mining holdout exposes the same distinction by difficulty. Each cell below is root MAE /
pairwise accuracy / top-move agreement against the 1M-node teacher:

| Evaluator | Easy | Medium | Deep | Unstable | Transitional |
|---|---|---|---|---|---|
| Handcrafted | .5473 / 55.68% / 46.50% | .4488 / 49.24% / 28.53% | .4282 / 48.94% / 21.50% | .3994 / 47.88% / 23.58% | .4894 / 51.04% / 32.54% |
| `phase-b-m` | .4856 / 67.99% / 48.85% | .4024 / 62.39% / 25.20% | .3806 / 64.02% / 18.67% | .3644 / 59.82% / 19.26% | .4502 / 62.83% / 31.70% |
| `phase-b-r-deep` | .4807 / 67.95% / 48.81% | .3932 / 62.85% / 25.62% | .3696 / 63.78% / 18.92% | .3573 / 59.65% / 19.34% | .4445 / 62.96% / 31.15% |
| `phase-b-h-medium` | .4820 / 67.90% / 48.05% | .3910 / 63.00% / 25.20% | .3699 / 63.62% / 17.50% | .3587 / 60.05% / 19.13% | .4449 / 62.82% / 31.29% |
| `phase-b-h-deep` | .4833 / 67.81% / 48.63% | .3936 / 62.99% / 25.69% | .3686 / 63.15% / 18.17% | .3582 / 59.48% / 20.27% | .4439 / 63.08% / 31.63% |
| `phase-b-m-fast` | .4917 / 66.84% / 48.51% | .4029 / 61.43% / 25.62% | .3845 / 62.86% / 19.83% | .3676 / 59.19% / 19.22% | .4513 / 62.09% / 31.73% |
| `phase-b-m-mid` | .4891 / 67.16% / 48.66% | .4060 / 61.44% / 25.76% | .3825 / 63.37% / 20.67% | .3673 / 58.94% / 19.26% | .4536 / 62.62% / 31.14% |

All four students improve value error and pairwise ordering in every bucket. Their exact top-move
advantage is confined to the easy bucket; all trail the handcrafted evaluator on the medium, deep,
unstable, and transitional buckets. This is consistent with the fixed-node advantage shrinking as
search grows.

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

| Candidate vs fallback | 1k nodes (320 games) | 10k nodes (80 games; `phase-b-m` 320) | 100k nodes (16 games) |
|---|---:|---:|---:|
| `medium` | 12.50% / -338 Elo | 15.63% / -293 Elo | 15.63% / -293 Elo |
| `combined` | 15.63% / -293 Elo | 3.13% / -597 Elo | 9.38% / -394 Elo |
| `fast-top` | 18.75% / -255 Elo | 0.00% / unbounded | 12.50% / -338 Elo |
| `corrected-64` | 12.50% / -338 Elo | not run | not run |
| `corrected-128` | 18.75% / -255 Elo | not run | not run |
| `phase-b-m` | **65.63% / +112 Elo** | 46.88% / -22 Elo | 43.75% / -44 Elo |
| `phase-b-r-deep` | 56.25% / +44 Elo | 50.00% / 0 Elo | 46.88% / -22 Elo |
| `phase-b-h-medium` | 50.00% / 0 Elo | 40.63% / -66 Elo | 28.13% / -163 Elo |
| `phase-b-h-deep` | 43.75% / -44 Elo | 37.50% / -89 Elo | 25.00% / -191 Elo |
| `phase-b-m-fast` | 37.50% / -89 Elo | stopped at gate | stopped at gate |
| `phase-b-m-mid` | 56.25% / +44 Elo | 46.88% / -22 Elo | not scheduled |

The first sweep uses more games at cheap node budgets and fewer games at 100k, then expands any
promising or ambiguous cell. Every cell disables clock enforcement, uses the same node ceiling on
both sides, and has zero failed terminations. The 100k cells are directional because they contain
only 16 games. `phase-b-m` is the first student to show clear fixed-search success: 65.63% over 320
games at 1k nodes, with a 61.88%-69.37% score interval. Its expanded 320-game 10k result is 46.88%
with a 42.34%-51.41% interval, statistically consistent with parity; its 100k result is directional.
Adding random-deep, selected-medium, or selected-deep examples weakens the 1k result in that order.
At this scale, medium-only breadth transfers to shallow search better than deeper or
disagreement-selected supervision.

The post-ablation `phase-b-m-fast` model preserves most one-ply metrics while reducing the artifact
from 4.47 MB to 1.17 MB, but it scores only 37.50% at 1k nodes over 320 games. Its larger-node cells
were stopped at the gate. Search-useful information is therefore sensitive to model capacity in a
way that evaluator MAE and pairwise accuracy do not reveal.

The 2.11 MB `phase-b-m-mid` midpoint recovers part of the lost search signal: it scores 56.25% at 1k
nodes over 320 games, approximately +44 Elo with a +14 to +74 Elo interval. The resulting capacity
frontier is +112 Elo for the 4.47 MB full model, +44 Elo for the 2.11 MB midpoint, and -89 Elo for
the 1.17 MB fast model. The midpoint is inconclusive at 10k (46.88% over 80 games), matching the
full model's initial result. Under the wall clock, however, the midpoint falls to 25.47%, compared
with 42.34% for full and 36.72% for fast. Capacity, one-ply accuracy, fixed-node strength, and
deployment strength are not monotonic across these architectures.

Two direct, color-paired 1k-node comparisons remove common-opponent noise from the primary causal
questions:

| Candidate | Opponent | W-D-L | Score | Elo | 95% score interval | Conclusion |
|---|---|---:|---:|---:|---:|---|
| `phase-b-h-deep` | `phase-b-h-medium` | 0-200-120 | 31.25% | -137 | 28.59%-33.91% | 1M labels hurt versus 100k labels on the same positions |
| `phase-b-h-deep` | `phase-b-r-deep` | 40-200-80 | 43.75% | -44 | 40.46%-47.04% | selected-hard positions hurt versus random positions at the same label depth |

Both comparisons use antisymmetric inference for both students and report zero failures. The
current data-mining hypothesis is rejected under this architecture and loss: deeper labels are
actively harmful, and disagreement selection is worse than random deep sampling.

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
| `phase-b-m` | fallback | 107-57-156 | 42.34% | -54 | 37.44%-47.25% | Large improvement, still reject |
| `phase-b-m-fast` | fallback | 78-79-163 | 36.72% | -95 | 32.19%-41.25% | Faster but loses search quality |
| `phase-b-m-mid` | fallback | 45-73-202 | 25.47% | -187 | 21.47%-29.47% | Fixed-node gain does not survive clock |

The first three benchmark reports were generated from independently verified commit `3de6dfe`.
Their embedded Git field is `unknown` because SSM ran as root against an `ec2-user` checkout. This
provenance bug was fixed in `c72dd2a`; new reports record the safe Git revision, model names, and
model SHA-256 hashes.

## AWS execution and guardrails

- Region: `us-east-1`; profile: `aichessathon`.
- Credential warning: the refreshed local profile currently resolves to the AWS account root.
  Replace it with an IAM Identity Center permission set before routine use; EC2 workers themselves
  already use a scoped instance role and no static credentials.
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
- Phase C corpus construction completed on one retained `c7i.8xlarge`. Its 9M medium labels are
  now running on eight `c7i.8xlarge` workers, again saturating the 256-vCPU Standard-instance
  quota. Four are retained workers and four are new disposable teacher workers.
- Pilot training used short `g6.xlarge` and `g5.xlarge` sessions. Phase B saturated one
  `g5.xlarge` A10G with four concurrent controlled models, then reused it for the two capacity
  variants.
- Phase C reuses the `g5.xlarge` A10G. The 9.0 MB, 19.3 MB, and 41.5 MB capacity cells completed;
  the largest cell used about 9.9 of 23.0 GiB GPU memory and remained below the 50 MB submission
  limit. The worker is configured to stop after its upload.
- Pilot benchmarks used disposable `c7i.2xlarge` workers. Phase B used four identical
  `c7i.8xlarge` workers for parallel fixed-node sweeps and a `c7i.4xlarge` of the same generation
  for wall-clock games; completed workers were reused for follow-up cells.
- All eight Phase B teacher workers are terminated. The retained Phase B training/evaluation
  workers are temporarily active for Phase C; one retained `c7i.4xlarge` remains stopped.
- Two untagged stopped instances created in April, each with an attached 16 GB `gp3` volume, remain
  in the account. They predate this project and were not modified.

The reservation ledger currently authorizes `$293.00` of worst-case monthly project spend,
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

## Phase C scaling study: in progress

Phase C is designed as two orthogonal axes so a data gain cannot be mistaken for a capacity gain:

| Axis | Controlled cells | Held constant |
|---|---|---|
| Broad medium data | 1M, 3M, 10M positions at 100k Stockfish nodes | 128/64/32 student, Phase B objective, batch, seed, and 20 full epochs |
| Student capacity | approximately 4.5, 10, 20, 40 MB on 1M positions | Phase B data and optimization recipe |

The 9M-position extension is disjoint at the evaluator-visible feature level from the pilot,
Phase B, and external evaluator corpora. It is divided into exact 2M and 7M components so every
larger cell is a strict superset of the smaller one. It uses 2,250 shards of 4,000 positions to
reduce the Phase B straggler penalty. Corpus generation completed on 2026-09-02 and eight CPU
workers are labeling the extension. The first 256 shards, or 1,024,000 new positions, uploaded
successfully. All three larger-capacity students completed. Pure evaluator and fixed 1k/10k-node
tests will select follow-ups; wall-clock is deliberately not the initial research gate.

The capacity cells below share the same 1M-position dataset, split, loss, seed, and number of
optimizer steps. These are training-validation diagnostics, not the independent evaluator test or
playing-strength results:

| Model | Architecture | Export size | Validation MSE | Validation ranking | Validation top move | Status |
|---|---|---:|---:|---:|---:|---|
| `phase-b-m` | 128/64/32 | 4.47 MB | .3132 | 63.42% | 34.71% | Existing control |
| `phase-c-d1m-c10` | 256/128/64 | 9.02 MB | .3073 | 64.01% | 34.86% | Complete |
| `phase-c-d1m-c20` | 512/256/128 | 19.33 MB | .3053 | 64.62% | 35.12% | Complete |
| `phase-c-d1m-c40` | 1024/512/256 | 41.53 MB | .3016 | 64.63% | 35.38% | Complete |

All three larger models improve every listed internal metric monotonically, but no scientific claim
is attached to that trend until they are evaluated against the untouched 1M-node holdout and in
color-paired fixed-node games. A 30M data cell is intentionally gated on the 10M curve: launch it if
the independent and fixed-node metrics are still improving rather than reserve that spend before
the 10M evidence exists.

The external calibration track will use DeepMind's Searchless Chess action-value checkpoints and
node-limited Stockfish in one local, color-paired opening pool. Any Elo values from that pool will
be reported only relative to its exact conditions. The main derived quantity is the Stockfish node
budget whose pool Elo matches a distilled-plus-search system, not a claimed transfer to human or
Lichess Elo.

## Decision gate and next work

The three experiments now support a specific diagnosis:

1. Distillation succeeds: every Phase B student improves unseen Stockfish value error and broad
   ordering, and `phase-b-m` slightly improves exact top-move agreement.
2. Medium-data scale creates useful search knowledge: `phase-b-m` is +112 Elo at a fixed 1k nodes.
   The benefit disappears by 10k, so it is shallow-search amortization rather than a universal
   evaluator upgrade.
3. The current deep-mining recipe is counterproductive. On direct paired tests, 1M labels lose to
   100k labels on the same hard positions, and selected-hard positions lose to random positions at
   the same label depth.
4. Deployment is still unsolved. Full, midpoint, and fast models all lose under the clock, and
   their wall-clock ordering is not explained by size or one-ply metrics alone.

The handcrafted evaluator remains the submission default. Do not spend the next compute tranche on
more labels from the current deep-selection recipe. For the research track, scale broad medium data
and student capacity until the pure-evaluator and fixed-node curves visibly saturate. In parallel,
prepare the Searchless Chess/Stockfish local reference pool. Only after identifying the strongest
research model should the deployment track prioritize quantization, incremental evaluation,
lower-level inference, caching, residual blending, or architecture changes under wall-clock.
