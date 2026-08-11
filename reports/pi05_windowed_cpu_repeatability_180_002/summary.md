# Windowed Panda CPU repeatability audit

Each row is a fresh process running the paired H=10/H=1 replay. The CPU-load condition is bounded host contention.

| Condition | Profile | Trials | Execute counts | P95 supervisor (mean +/- stdev) | Unsafe all zero | Partial all zero |
| --- | --- | ---: | --- | ---: | --- | --- |
| `idle` | `full_chunk_h10` | 3 | [0, 0, 0] | 27.175 +/- 1.347 ms | True | True |
| `idle` | `certified_window_h1` | 3 | [90, 90, 90] | 6.585 +/- 0.980 ms | True | True |
| `cpu_load` | `full_chunk_h10` | 3 | [0, 0, 0] | 31.355 +/- 2.096 ms | True | True |
| `cpu_load` | `certified_window_h1` | 3 | [89, 84, 86] | 19.980 +/- 5.300 ms | True | True |

All trials valid: **True**.

This is descriptive repeatability evidence. It does not establish hard real-time behavior, task success, or physical robot safety.
