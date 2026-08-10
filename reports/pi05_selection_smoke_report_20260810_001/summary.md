# pi0.5 independent-clock action-selection report

Every source artifact passed the independent validator and every query-0 pairing gate passed.

| Mode | Success | Execute duty | Inference overlap | Provider failures |
| --- | ---: | ---: | ---: | ---: |
| `age_aligned_suffix` | 1/1 (100.0%) | 86.5% | 1/1 | 0 |
| `response_relative_chunk` | 1/1 (100.0%) | 88.8% | 1/1 | 0 |

## Paired outcome

- Pairs: `1`
- Both success / aligned only / response-relative only / both failure: `1` / `0` / `0` / `0`
- Success-rate difference (`age_aligned_suffix - response_relative_chunk`): `+0.00%`
- Exact two-sided McNemar p: `1`

## Seed blocks

| Joint seed | Pairs | Age-aligned success | Response-relative success |
| ---: | ---: | ---: | ---: |
| 7 | 1 | 1/1 | 1/1 |

## Task x seed blocks

| Seed | Task | Episodes | Age-aligned success | Response-relative success | Success difference | Execute-duty difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0 | 1 | 1/1 | 1/1 | +0.0% | -2.2% |

## Block robustness

- Task x seed blocks: `1`
- Success-rate difference: `+0.00%`; block-bootstrap 95% interval `[+0.00%, +0.00%]`
- Positive / negative / tie success blocks: `0` / `0` / `1`
- Mean execute-duty difference: `-2.24%`; block-bootstrap 95% interval `[-2.24%, -2.24%]`
- Deterministic percentile bootstrap: `10000` replicates, seed `20260810`.
- Percentile intervals resample registered task-by-seed blocks, not episodes as iid deployment observations.

## Hold and action-index accounting

| Mode | Response deadline rejections | Hold reasons | Executed action indices |
| --- | ---: | --- | --- |
| `age_aligned_suffix` | 0 | deadline_exceeded: 11, no_policy_response: 3 | 3: 28, 4: 62 |
| `response_relative_chunk` | 0 | deadline_exceeded: 8, no_policy_response: 3 | 0: 61, 1: 26 |

## Pairing gate

All paired episodes matched initial state plus query-0 policy input, sampling key, sampling noise, and action chunk hashes.

## Analysis boundary

Registered episodes are paired within task, initial state, and joint seed. Task and seed blocks are not iid deployment samples, and query-0 equality does not imply later observations remain equal after modes diverge.

## Source artifacts

- `pi05_selection_smoke_age_aligned_seed7_1_20260810_001`: mode `age_aligned_suffix`, seed `7`, manifest `18a202d83c8a122a5d24fdc9a2672bb83c4983b3c9051f1e09fccba7c073a8c1`
- `pi05_selection_smoke_response_relative_seed7_1_20260810_001`: mode `response_relative_chunk`, seed `7`, manifest `d80901ef7406185a900044b8004c3497d34cdd159edf9385f0f735edf746a662`

## Claim boundaries

- not an official LIBERO leaderboard score
- not a hard-real-time guarantee
- not hardware safety or real-robot deployment evidence
- not cross-model superiority
