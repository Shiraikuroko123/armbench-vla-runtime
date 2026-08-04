# Measured-age paired analysis

- Source manifest SHA-256: `ce439762242328ceeee5177c470ee498becfe422beaeec868a9b2637eddc8d39`
- Rollouts/pairs/queries: 240/120/4547
- Bootstrap seed/resamples: `20260805/10000`

| Mode | Success (Wilson 95%) | Age P95 / max ms | Deadline | Horizon | Refresh |
| --- | ---: | ---: | ---: | ---: | ---: |
| async_unguarded | 88/120 [0.648, 0.804] | 248.540 / 263.976 | 48 | 48 | 0 |
| latency_aligned | 116/120 [0.917, 0.987] | 248.403 / 254.778 | 25 | 25 | 25 |

Paired aligned-minus-async success difference: **+0.233** (bootstrap 95% [+0.150, +0.317]); wins/losses/ties 32/4/84; exact McNemar p=1.94157e-06.

Simulation-only measured-age analysis. Inference remains blocking and controller catch-up is simulated after response arrival.
