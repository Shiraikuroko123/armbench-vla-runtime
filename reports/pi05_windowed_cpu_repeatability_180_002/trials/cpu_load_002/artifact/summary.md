# Windowed Panda CPU assurance audit

The same frozen pi0.5-to-Panda inputs are evaluated as a complete H=10 publication and as an H=1 certified execution window.

| Profile | Cases | Execute | Constraint-safe | Unsafe windows | Partial windows | Budget misses | P95 supervisor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_chunk_h10` | 90 | 0 | 66 | 0 | 0 | 85 | 30.229 ms |
| `certified_window_h1` | 90 | 84 | 90 | 0 | 0 | 6 | 27.085 ms |

Window minus full-chunk execute count: **+84**.

The H=1 result changes publication granularity. It does not claim full-source-chunk atomicity, task success, hard real-time execution, or physical safety.
