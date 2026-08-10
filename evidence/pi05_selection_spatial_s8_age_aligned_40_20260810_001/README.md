# pi0.5 independent-clock selection study: seed 8 age-aligned

This is the registered seed-8 `age_aligned_suffix` cell in the frozen
action-selection study. It covers LIBERO Spatial tasks 0-9, held-out episodes
4-7, a 175 ms observation-age deadline, and the official `pi05_libero`
checkpoint. Independent clocks, latest-only mailbox, keyed sampling, fail-closed
hold semantics, and ArmBench runtime commit
`1551900d2c66b0e8a1d46af51ee5df53e8c63bcc` are fixed by the protocol.

| Measure | Value |
| --- | ---: |
| Completed rollouts | 40 / 40 |
| Task success | 36 / 40 (90.0%) |
| Inference/simulation overlap | 40 / 40 |
| Execute / hold ticks | 4,128 / 624 |
| Execute duty cycle | 86.9% |
| Response deadline rejections / provider failures | 0 / 0 |

The independently rebuilt seed-8 pair report verified all 40 query-0 pairs
against the response-relative cell. It found 25 both-success, 11 aligned-only,
4 response-relative-only, and 0 both-failure outcomes. The seed-only paired
difference is +17.5 percentage points with exact two-sided McNemar `p=0.1185`.

```powershell
& '.\.venv\Scripts\python.exe' -m integrations.openpi.validate_libero_independent_clock `
  '.\evidence\pi05_selection_spatial_s8_age_aligned_40_20260810_001\evaluation' --json
```

The verified transport archive SHA-256 is in `transport_sha256.txt`.

This is one seed block in an exploratory held-out simulation comparison. It is
not a standalone superiority result, official leaderboard score,
hard-real-time guarantee, hardware-safety result, cross-model comparison, or
iid deployment estimate. Query-0 equality does not imply that observations
remain equal after the two modes produce different trajectories.
