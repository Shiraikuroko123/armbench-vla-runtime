# pi0.5-LIBERO measured-age evaluation

- Schema: `armbench.pi05_libero_measured_age.v2`
- Planned/completed rollouts: 40/40
- Non-scoring warm-up queries: 3
- Complete: yes

| Mode | Success | Queries | Age P95 ms | Deadline misses | Horizon overruns |
| --- | ---: | ---: | ---: | ---: | ---: |
| async_unguarded | 14/20 | 492 | 244.740 | 1 | 1 |
| latency_aligned | 19/20 | 318 | 244.587 | 1 | 1 |

This artifact uses blocking inference plus post-response simulator catch-up. It is not an OS hard-real-time or real-robot result.
