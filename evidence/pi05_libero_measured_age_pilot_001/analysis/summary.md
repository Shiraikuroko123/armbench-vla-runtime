# Measured-age paired analysis

- Source manifest SHA-256: `fb233967926e78ba26231556f70e1835909273e58813b0761e63a2f7f70bf02a`
- Rollouts/pairs/queries: 40/20/810
- Bootstrap seed/resamples: `20260805/10000`

| Mode | Success (Wilson 95%) | Age P95 / max ms | Deadline | Horizon | Refresh |
| --- | ---: | ---: | ---: | ---: | ---: |
| async_unguarded | 14/20 [0.481, 0.855] | 244.740 / 250.166 | 1 | 1 | 0 |
| latency_aligned | 19/20 [0.764, 0.991] | 244.587 / 251.428 | 1 | 1 | 1 |

Paired aligned-minus-async success difference: **+0.250** (bootstrap 95% [+0.100, +0.450]); wins/losses/ties 5/0/15; exact McNemar p=0.0625.

Simulation-only measured-age analysis. Inference remains blocking and controller catch-up is simulated after response arrival.
