# Frozen pi0.5 to integrated Panda CPU replay

The official checkpoint was not executed in this run. This report replays attested frozen responses and does not measure task success.

| Mode | Cases | Execute | Hold | Constraint-safe candidates | Unsafe published | Budget misses | P95 mode latency | Motion retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_dispatch` | 90 | 90 | 0 | 0 | 90 | 0 | 0.002 ms | 1.000 |
| `qp_projection` | 90 | 1 | 89 | 5 | 1 | 89 | 28.848 ms | 0.009 |
| `full_assurance` | 90 | 0 | 90 | 5 | 0 | 90 | 42.974 ms | 0.000 |

Registered go/no-go result: **no_go**.

Hand displacement is only a command-retention proxy. It is not LIBERO or Panda task progress. Timing is measured best-effort Python CPU timing, not a worst-case real-time guarantee.
