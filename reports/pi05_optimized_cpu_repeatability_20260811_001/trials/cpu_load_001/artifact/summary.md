# Optimized pi0.5-to-Panda CPU replay

Frozen responses are replayed through the optimized atomic runtime; the checkpoint is not executed.

| Profile | Cases | Execute | Constraint-safe | Unsafe published | Budget misses | P95 supervisor | P95 worker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `operational_20ms` | 90 | 0 | 66 | 0 | 85 | 27.892 ms | 32.000 ms |
| `diagnostic_100ms` | 90 | 6 | 66 | 0 | 44 | 116.667 ms | 118.250 ms |

Operational go/no-go: **no_go**.

The 100 ms profile is diagnostic only and cannot change the 20 ms decision.
