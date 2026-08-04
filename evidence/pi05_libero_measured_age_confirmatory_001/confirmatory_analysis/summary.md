# Measured-age confirmatory task-cluster analysis

- Source manifest SHA-256: `ce439762242328ceeee5177c470ee498becfe422beaeec868a9b2637eddc8d39`
- Tasks/pairs/rollouts: 10/120/240
- Primary pooled exact McNemar: wins/losses/ties 32/4/84, risk difference +0.2333, two-sided p=1.9415747e-06
- Whole-task bootstrap: +0.1083 to +0.3833 (10000 resamples, seed 20260805)
- Exact task sign-flip: 16/1024 extreme assignments, p=0.015625
- Leave-one-task-out risk-difference range: +0.1759 to +0.2593

| Task | Async | Aligned | Wins / losses / ties | Risk difference |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8/12 | 12/12 | 4 / 0 / 8 | +0.3333 |
| 1 | 11/12 | 12/12 | 1 / 0 / 11 | +0.0833 |
| 2 | 12/12 | 12/12 | 0 / 0 / 12 | +0.0000 |
| 3 | 9/12 | 11/12 | 3 / 1 / 8 | +0.1667 |
| 4 | 7/12 | 12/12 | 5 / 0 / 7 | +0.4167 |
| 5 | 3/12 | 12/12 | 9 / 0 / 3 | +0.7500 |
| 6 | 12/12 | 12/12 | 0 / 0 / 12 | +0.0000 |
| 7 | 11/12 | 11/12 | 1 / 1 / 10 | +0.0000 |
| 8 | 8/12 | 12/12 | 4 / 0 / 8 | +0.3333 |
| 9 | 7/12 | 10/12 | 5 / 2 / 5 | +0.2500 |

| Condition first | Pairs | Async | Aligned | Wins / losses / ties | Risk difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| async_unguarded | 60 | 50/60 | 58/60 | 10 / 2 / 48 | +0.1333 |
| latency_aligned | 60 | 38/60 | 58/60 | 22 / 2 / 36 | +0.3333 |

Simulation-only task-cluster robustness analysis of a validated measured-age pi0.5-LIBERO artifact.
