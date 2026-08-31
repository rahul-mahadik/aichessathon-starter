# Benchmarking

`benchmarks.run` plays paired games from every position in `openings.epd`: once with the candidate
as White and once as Black. This removes most colour and opening bias. The suite is intentionally
public and small; it is not intended to imitate the competition's unpublished opening set.

```bash
make bench
make bench BENCH_OPPONENT=../previous-agent BENCH_ROUNDS=10 BENCH_BASE_MS=30000
uv run python -m benchmarks.run --help
```

On a multi-core AWS worker, pass `--workers N`. Each match launches two single-threaded agents, so
leave physical CPU headroom rather than setting workers equal to the advertised vCPU count.

Results include every game's termination, elapsed wall time and PGN, plus the current Git revision.
They are written to `benchmark-results/`, which is ignored by Git. Keep named result files outside
that directory if they should become permanent experiment records.

For engine-strength decisions, compare against a frozen previous version, alternate colours, and
run hundreds of games. Eight positions and one round are only a correctness smoke test.
