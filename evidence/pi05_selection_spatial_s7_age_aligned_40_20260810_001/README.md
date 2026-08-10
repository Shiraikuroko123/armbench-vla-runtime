# pi0.5 independent-clock selection study: seed 7 age-aligned

This is the first registered scored cell in the frozen action-selection study.
It covers LIBERO Spatial tasks 0-9, held-out episodes 4-7, joint seed 7, a
175 ms observation-age deadline, and `age_aligned_suffix` selection. The
checkpoint, independent clocks, latest-only mailbox, keyed sampling, hold
fallback, and runtime commit are fixed by the protocol.

| Measure | Value |
| --- | ---: |
| Completed rollouts | 40 / 40 |
| Task success | 38 / 40 (95.0%) |
| Inference/simulation overlap | 40 / 40 |
| Execute / hold ticks | 3,942 / 667 |
| Execute duty cycle | 85.5% |
| Response deadline rejections / provider failures | 0 / 0 |

The independently rebuilt seed-7 pair report verified all 40 query-0 pairs
against the response-relative cell and found 34 both-success, 4 aligned-only,
2 response-relative-only, and 0 both-failure outcomes. The seed-only paired
difference was +5.0 percentage points with exact two-sided McNemar `p=0.6875`.

```powershell
& '.\.venv\Scripts\python.exe' -m integrations.openpi.validate_libero_independent_clock `
  '.\evidence\pi05_selection_spatial_s7_age_aligned_40_20260810_001\evaluation' --json
```

The verified transport archive SHA-256 is in `transport_sha256.txt`.

This is one seed block in an exploratory held-out simulation comparison. It is
not a standalone superiority result, official leaderboard score,
hard-real-time guarantee, hardware-safety result, or cross-model comparison.
