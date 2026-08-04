# pi0.5-LIBERO asynchronous runtime evaluation

This artifact reports official LIBERO task completion (`done`). It does not interpret object contact as a safety violation.

- Planned rollouts: 2
- Completed rollouts: 2
- Successful rollouts: 2
- Non-efficacy runtime/contract failures retained: 0
- Artifact schema: `armbench.pi05_libero_async.v1`

## Aggregate conditions

| Scope | Task | Mode | Horizon | Delay steps | Refresh N | PP/ITT N | ITT success | ITT 95% Wilson CI | PP success | Queries | Rejections |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| libero_spatial | all | async_unguarded | 5 | 0 | n/a | 2/2 | 1.000 | [0.342, 1.000] | 1.000 | 21.50 | 0.00 |

## Paired comparisons

| Scope | Horizon | Delay steps | Pairs | Guard - unguarded | Bootstrap 95% CI | Holm p | Query overhead |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |

Results with small N are smoke-test evidence only. A competitive claim requires the preregistered paired sample count, all negative results, and the pinned checkpoint recorded in `resolved_protocol.json`.

## Artifact integrity

- Valid: yes
