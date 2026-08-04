# pi0.5-LIBERO measured-age evaluation

- Schema: `armbench.pi05_libero_measured_age.v2`
- Planned/completed rollouts: 240/240
- Non-scoring warm-up queries: 3
- Complete: yes

| Mode | Success | Queries | Age P95 ms | Deadline misses | Horizon overruns |
| --- | ---: | ---: | ---: | ---: | ---: |
| async_unguarded | 88/120 | 2730 | 248.540 | 48 | 48 |
| latency_aligned | 116/120 | 1817 | 248.403 | 25 | 25 |

This artifact uses blocking inference plus post-response simulator catch-up. It is not an OS hard-real-time or real-robot result.
