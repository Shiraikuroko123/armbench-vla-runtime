# pi0.5-LIBERO asynchronous runtime evaluation

This artifact reports official LIBERO task completion (`done`). It does not interpret object contact as a safety violation.

- Planned rollouts: 40
- Completed rollouts: 40
- Successful rollouts: 13
- Non-efficacy runtime/contract failures retained: 0
- Artifact schema: `armbench.pi05_libero_async.v1`

## Aggregate conditions

| Scope | Task | Mode | Horizon | Delay steps | Refresh N | PP/ITT N | ITT success | ITT 95% Wilson CI | PP success | Queries | Rejections |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| libero_spatial | all | async_unguarded | 5 | 4 | n/a | 20/20 | 0.350 | [0.181, 0.567] | 0.350 | 23.00 | 0.00 |
| libero_spatial | all | state_guard | 5 | 4 | n/a | 20/20 | 0.300 | [0.145, 0.519] | 0.300 | 24.10 | 14.10 |

## Paired comparisons

| Scope | Horizon | Delay steps | Pairs | Guard - unguarded | Bootstrap 95% CI | Holm p | Query overhead |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| libero_spatial | 5 | 4 | 20 | -0.050 | [-0.300, 0.200] | 1.0000 | 4.8% |

Results with small N are smoke-test evidence only. A competitive claim requires the preregistered paired sample count, all negative results, and the pinned checkpoint recorded in `resolved_protocol.json`.

## Artifact integrity

- Valid: yes
