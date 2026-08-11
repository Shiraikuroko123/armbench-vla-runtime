# Windowed Panda CPU assurance audit

The same frozen pi0.5-to-Panda inputs are evaluated as a complete H=10 publication and as an H=1 certified execution window.

| Profile | Cases | Execute | Constraint-safe | Unsafe windows | Partial windows | Budget misses | P95 supervisor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_chunk_h10` | 90 | 0 | 66 | 0 | 0 | 86 | 29.544 ms |
| `certified_window_h1` | 90 | 86 | 90 | 0 | 0 | 4 | 18.495 ms |

Window minus full-chunk execute count: **+86**.

The H=1 result changes publication granularity. It does not claim full-source-chunk atomicity, task success, hard real-time execution, or physical safety.
