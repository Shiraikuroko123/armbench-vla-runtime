# Optimized CPU repeatability audit

Each row is one fresh Python process running the frozen 180-case optimized replay.
The CPU-load condition is a bounded host-contention diagnostic.

| Condition | Profile | Trials | Execute counts | P95 supervisor (mean +/- stdev) | P95 worker (mean +/- stdev) | Unsafe all zero | Prefixes all zero |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| `idle` | `operational_20ms` | 3 | [1, 0, 0] | 27.041 +/- 1.078 ms | 32.000 +/- 0.000 ms | True | True |
| `idle` | `diagnostic_100ms` | 3 | [20, 39, 16] | 109.469 +/- 0.951 ms | 109.667 +/- 0.471 ms | True | True |
| `cpu_load` | `operational_20ms` | 3 | [0, 0, 0] | 28.385 +/- 1.025 ms | 32.000 +/- 0.000 ms | True | True |
| `cpu_load` | `diagnostic_100ms` | 3 | [6, 5, 5] | 116.095 +/- 0.654 ms | 117.750 +/- 6.134 ms | True | True |

All trials valid: **True**.

The descriptive repeatability result does not establish hard real-time behavior, task success, or physical-robot safety.
