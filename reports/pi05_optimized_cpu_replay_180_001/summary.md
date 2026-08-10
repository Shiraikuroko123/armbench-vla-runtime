# Optimized pi0.5-to-Panda CPU replay

Frozen responses are replayed through the optimized atomic runtime; the checkpoint is not executed.

| Profile | Cases | Execute | Constraint-safe | Unsafe published | Budget misses | P95 supervisor | P95 worker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `operational_20ms` | 90 | 1 | 66 | 0 | 69 | 23.888 ms | 32.000 ms |
| `diagnostic_100ms` | 90 | 58 | 66 | 0 | 3 | 86.559 ms | 86.700 ms |

Operational go/no-go: **go**.

The 100 ms profile is diagnostic only and cannot change the 20 ms decision.
