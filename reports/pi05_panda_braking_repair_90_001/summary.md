# Braking-invariant repair on frozen pi0.5 responses

This paired offline diagnostic compares the existing greedy guard
with a trajectory-scale search that preserves a collision-valid
terminal braking path. The checkpoint was not executed in this run.

## Paired outcome

| Metric | Greedy guard | Braking-invariant repair |
| --- | ---: | ---: |
| All registered constraints satisfied | 264 / 270 | 270 / 270 |
| Position path valid | 270 / 270 | 270 / 270 |
| Acceleration-conflict cases | 6 | 0 |
| P95 software latency | 13.550 ms | 12.777 ms |

Resolved legacy conflicts: 6
Repair regressions: 0
Selected trajectory scales: 1: 233, 0.75: 3, 0.5: 9, 0.25: 21, 0: 4
Selection-deadline exceedances: 0

## Claim boundary

No task success, Panda closed loop, hard-real-time guarantee,
continuous-collision certificate, or physical-safety claim is made.
