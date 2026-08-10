# pi0.5 independent-clock selection study: seed 9 response-relative

This is the registered seed-9 `response_relative_chunk` cell in the frozen
action-selection study. It covers LIBERO Spatial tasks 0-9, held-out episodes
4-7, a 175 ms observation-age deadline, and the official `pi05_libero`
checkpoint. The selector starts accepted responses at action index zero while
retaining the same deadline and fail-closed hold rule. Independent clocks,
latest-only mailbox, keyed sampling, and ArmBench runtime commit
`1551900d2c66b0e8a1d46af51ee5df53e8c63bcc` are fixed by the protocol.

| Measure | Value |
| --- | ---: |
| Completed rollouts | 40 / 40 |
| Task success | 35 / 40 (87.5%) |
| Inference/simulation overlap | 40 / 40 |
| Execute / hold ticks | 5,469 / 781 |
| Execute duty cycle | 87.5% |
| Response deadline rejections / provider failures | 0 / 0 |

The seed-9 paired block verified all 40 query-0 initial-state, policy-input,
sampling-key, sampling-noise, and action-chunk hashes. It found 35 both-success,
5 aligned-only, 0 response-relative-only, and 0 both-failure outcomes. The
seed-only paired difference is +12.5 percentage points with exact two-sided
McNemar `p=0.0625`.

```powershell
& '.\.venv\Scripts\python.exe' -m integrations.openpi.validate_libero_independent_clock `
  '.\evidence\pi05_selection_spatial_s9_response_relative_40_20260810_001\evaluation' --json
```

The verified transport archive SHA-256 is in `transport_sha256.txt`. Interpret
this artifact through the registered 120-pair report, not as a standalone seed
result.

This is not an unguarded baseline. It is one block in an exploratory held-out
simulation comparison, not an official leaderboard score, hard-real-time
guarantee, hardware-safety result, cross-model comparison, or iid deployment
estimate. Query-0 equality does not imply later observations remain equal after
the modes diverge.
