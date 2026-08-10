# pi0.5 independent-clock action-selection report

Every source artifact passed the independent validator and every query-0 pairing gate passed.

| Mode | Success | Execute duty | Inference overlap | Provider failures |
| --- | ---: | ---: | ---: | ---: |
| `age_aligned_suffix` | 114/120 (95.0%) | 86.4% | 120/120 | 0 |
| `response_relative_chunk` | 100/120 (83.3%) | 87.2% | 120/120 | 0 |

## Paired outcome

- Pairs: `120`
- Both success / aligned only / response-relative only / both failure: `94` / `20` / `6` / `0`
- Success-rate difference (`age_aligned_suffix - response_relative_chunk`): `+11.67%`
- Exact two-sided McNemar p: `0.0093553066`

## Seed blocks

| Joint seed | Pairs | Age-aligned success | Response-relative success |
| ---: | ---: | ---: | ---: |
| 7 | 40 | 38/40 | 36/40 |
| 8 | 40 | 36/40 | 29/40 |
| 9 | 40 | 40/40 | 35/40 |

## Task x seed blocks

| Seed | Task | Episodes | Age-aligned success | Response-relative success | Success difference | Execute-duty difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0 | 4 | 4/4 | 4/4 | +0.0% | -0.0% |
| 7 | 1 | 4 | 4/4 | 3/4 | +25.0% | -1.1% |
| 7 | 2 | 4 | 4/4 | 4/4 | +0.0% | -4.3% |
| 7 | 3 | 4 | 4/4 | 4/4 | +0.0% | -2.4% |
| 7 | 4 | 4 | 3/4 | 4/4 | -25.0% | -1.8% |
| 7 | 5 | 4 | 4/4 | 2/4 | +50.0% | -1.7% |
| 7 | 6 | 4 | 3/4 | 4/4 | -25.0% | -0.4% |
| 7 | 7 | 4 | 4/4 | 4/4 | +0.0% | -1.0% |
| 7 | 8 | 4 | 4/4 | 4/4 | +0.0% | +0.0% |
| 7 | 9 | 4 | 4/4 | 3/4 | +25.0% | -1.5% |
| 8 | 0 | 4 | 4/4 | 2/4 | +50.0% | -1.2% |
| 8 | 1 | 4 | 4/4 | 3/4 | +25.0% | +0.1% |
| 8 | 2 | 4 | 4/4 | 4/4 | +0.0% | +0.0% |
| 8 | 3 | 4 | 4/4 | 3/4 | +25.0% | -2.1% |
| 8 | 4 | 4 | 4/4 | 4/4 | +0.0% | -0.1% |
| 8 | 5 | 4 | 4/4 | 1/4 | +75.0% | -0.6% |
| 8 | 6 | 4 | 4/4 | 4/4 | +0.0% | -0.6% |
| 8 | 7 | 4 | 1/4 | 4/4 | -75.0% | +1.3% |
| 8 | 8 | 4 | 4/4 | 2/4 | +50.0% | -0.4% |
| 8 | 9 | 4 | 3/4 | 2/4 | +25.0% | +0.8% |
| 9 | 0 | 4 | 4/4 | 4/4 | +0.0% | +0.8% |
| 9 | 1 | 4 | 4/4 | 3/4 | +25.0% | -1.0% |
| 9 | 2 | 4 | 4/4 | 4/4 | +0.0% | +0.4% |
| 9 | 3 | 4 | 4/4 | 4/4 | +0.0% | +0.0% |
| 9 | 4 | 4 | 4/4 | 4/4 | +0.0% | -0.2% |
| 9 | 5 | 4 | 4/4 | 2/4 | +50.0% | -1.7% |
| 9 | 6 | 4 | 4/4 | 4/4 | +0.0% | -0.5% |
| 9 | 7 | 4 | 4/4 | 4/4 | +0.0% | -1.5% |
| 9 | 8 | 4 | 4/4 | 4/4 | +0.0% | -1.1% |
| 9 | 9 | 4 | 4/4 | 2/4 | +50.0% | -0.7% |

## Block robustness

- Task x seed blocks: `30`
- Success-rate difference: `+11.67%`; block-bootstrap 95% interval `[+1.67%, +21.67%]`
- Positive / negative / tie success blocks: `12` / `3` / `15`
- Mean execute-duty difference: `-0.75%`; block-bootstrap 95% interval `[-1.16%, -0.38%]`
- Deterministic percentile bootstrap: `10000` replicates, seed `20260810`.
- Percentile intervals resample registered task-by-seed blocks, not episodes as iid deployment observations.

## Hold and action-index accounting

| Mode | Response deadline rejections | Hold reasons | Executed action indices |
| --- | ---: | --- | --- |
| `age_aligned_suffix` | 0 | deadline_exceeded: 1516, no_policy_response: 359 | 3: 3630, 4: 8303 |
| `response_relative_chunk` | 2 | deadline_exceeded: 2088, no_policy_response: 359 | 0: 11803, 1: 4837 |

## Pairing gate

All paired episodes matched initial state plus query-0 policy input, sampling key, sampling noise, and action chunk hashes.

## Analysis boundary

Registered episodes are paired within task, initial state, and joint seed. Task and seed blocks are not iid deployment samples, and query-0 equality does not imply later observations remain equal after modes diverge.

## Frozen matrix gate

Profile `frozen-240` passed: runtime commit `1551900d2c66b0e8a1d46af51ee5df53e8c63bcc`, seeds `7, 8, 9`, two modes, LIBERO-Spatial tasks `0-9`, episodes `4-7`, and `120 pairs / 240 rollouts`.

## Source artifacts

- `pi05_selection_spatial_s7_age_aligned_40_20260810_001`: mode `age_aligned_suffix`, seed `7`, manifest `8adee298bdfe4d5f34078772eacfbb19ebb0e1a8ffd391a6402ec17613d6923f`
- `pi05_selection_spatial_s7_response_relative_40_20260810_001`: mode `response_relative_chunk`, seed `7`, manifest `a1e95e6ffdeaf295ff4cf087f1c5dcf100c194d2df020bb4e78abdc863b85380`
- `pi05_selection_spatial_s8_age_aligned_40_20260810_001`: mode `age_aligned_suffix`, seed `8`, manifest `24e30103f58748e17e5e1885066e1e0c93d6ed9977981af5adbd1e872dc78cbc`
- `pi05_selection_spatial_s8_response_relative_40_20260810_001`: mode `response_relative_chunk`, seed `8`, manifest `1db070a6b626eb720088f816cf891792146d74a3759d5f530e7cf390b66498bf`
- `pi05_selection_spatial_s9_age_aligned_40_20260810_001`: mode `age_aligned_suffix`, seed `9`, manifest `65dfce297164e508f3bbe23bca872382f71b61ed8b91264b7e55a5cba06a71f5`
- `pi05_selection_spatial_s9_response_relative_40_20260810_001`: mode `response_relative_chunk`, seed `9`, manifest `706fad8a91b2a348ccae42072e86ae4447a859a3ecca8a9e91d12369e5a27eef`

## Claim boundaries

- not an official LIBERO leaderboard score
- not a hard-real-time guarantee
- not hardware safety or real-robot deployment evidence
- not cross-model superiority
