# pi0.5-LIBERO deadline curve (exploratory)

This report joins four separately preserved independent-clock MuJoCo artifacts
that use the same official `pi05_libero` checkpoint, Spatial task matrix, seed,
20 Hz control period, latest-only mailbox, and fail-closed hold rule. Only the
response deadline changes. The table is descriptive evidence for runtime
budgeting, not a confirmatory threshold analysis.

| Deadline | Rollouts | Task success | Execute ticks | Hold ticks | Deadline exceedances | Execute duty cycle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 ms | 40 | 0 / 40 (0.0%) | 0 | 8,800 | 5,474 | 0.0% |
| 150 ms | 40 | 0 / 40 (0.0%) | 2,309 | 6,491 | 16 | 26.2% |
| 175 ms | 40 | 38 / 40 (95.0%) | 3,942 | 613 | 0 | 86.5% |
| 200 ms | 40 | 38 / 40 (95.0%) | 4,031 | 592 | 0 | 87.2% |

The 150-to-175 ms change is a transition in this particular service and task
matrix: action execution becomes frequent enough for task progress to return.
It must not be generalized to a universal VLA deadline, OS hard-real-time
guarantee, hardware safety margin, or model-quality ranking. G04, G05, and G06
remain exploratory artifacts and are not pooled into a formal effect estimate.

## Source artifacts

- [G04 50 ms](../../evidence/g04_spatial_deadline50_40_20260810_001/README.md)
- [G05 150 ms](../../evidence/g05_spatial_deadline150_40_20260810_001/README.md)
- [G06 175 ms](../../evidence/g06_spatial_deadline175_40_20260810_001/README.md)
- [G02 200 ms](../../evidence/pi05_libero_independent_clock_core_40_001/summary.md)
