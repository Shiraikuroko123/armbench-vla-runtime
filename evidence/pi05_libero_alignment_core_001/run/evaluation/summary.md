# pi0.5-LIBERO asynchronous runtime evaluation

This artifact reports official LIBERO task completion (`done`). It does not interpret object contact as a safety violation.

- Planned rollouts: 300
- Completed rollouts: 300
- Successful rollouts: 256
- Non-efficacy runtime/contract failures retained: 0
- Artifact schema: `armbench.pi05_libero_async.v1`

## Aggregate conditions

| Scope | Task | Mode | Horizon | Delay steps | Refresh N | PP/ITT N | ITT success | ITT 95% Wilson CI | PP success | Queries | Rejections |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| libero_spatial | all | async_unguarded | 5 | 0 | n/a | 50/50 | 0.980 | [0.895, 0.996] | 0.980 | 22.14 | 0.00 |
| libero_spatial | all | async_unguarded | 5 | 2 | n/a | 50/50 | 0.820 | [0.692, 0.902] | 0.820 | 21.80 | 0.00 |
| libero_spatial | all | async_unguarded | 5 | 4 | n/a | 50/50 | 0.360 | [0.241, 0.499] | 0.360 | 22.96 | 0.00 |
| libero_spatial | all | latency_aligned | 5 | 0 | n/a | 50/50 | 1.000 | [0.929, 1.000] | 1.000 | 21.50 | 0.00 |
| libero_spatial | all | latency_aligned | 5 | 2 | n/a | 50/50 | 0.960 | [0.865, 0.989] | 0.960 | 16.18 | 0.00 |
| libero_spatial | all | latency_aligned | 5 | 4 | n/a | 50/50 | 1.000 | [0.929, 1.000] | 1.000 | 12.82 | 0.00 |

## Paired comparisons

| Scope | Horizon | Delay steps | Pairs | Guard - unguarded | Bootstrap 95% CI | Holm p | Query overhead |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |

Results with small N are smoke-test evidence only. A competitive claim requires the preregistered paired sample count, all negative results, and the pinned checkpoint recorded in `resolved_protocol.json`.

## Artifact integrity

- Valid: yes
