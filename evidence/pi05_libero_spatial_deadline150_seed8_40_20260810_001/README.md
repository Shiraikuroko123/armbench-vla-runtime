# pi0.5-LIBERO Spatial 150 ms seed-8 replication

This artifact repeats the registered 40-rollout Spatial deadline cell with the
joint environment and keyed-policy sampling seed changed from 7 to 8. The task
matrix, checkpoint, 20 Hz control clock, latest-only mailbox, fail-closed hold
rule, and 150 ms deadline remain fixed. It is an exploratory second-seed
replication selected after observing the seed-7 result, not a confirmatory
study.

## Result

| Measure | Value |
| --- | ---: |
| Completed rollouts | 40 / 40 |
| LIBERO task success | 0 / 40 (0.0%) |
| Episodes with measured inference overlap | 40 / 40 |
| Control ticks | 8,800 |
| Ticks during inference | 8,678 |
| Execute / hold ticks | 2,342 / 6,458 |
| Tick-level deadline holds | 6,338 |
| Response-level deadline rejections / provider failures | 2 / 0 |

Tick-level deadline holds and response-level deadline rejections are distinct:
one rejected or stale response can affect multiple 20 Hz control ticks. The
seed-8 result reproduces the low execute duty cycle and zero task successes
observed at 150 ms for seed 7.

## Independent validation

From the repository root on Windows:

```powershell
& '.\.venv\Scripts\python.exe' -m integrations.openpi.validate_libero_independent_clock `
  '.\evidence\pi05_libero_spatial_deadline150_seed8_40_20260810_001\evaluation' --json
```

The validator recomputes manifest hashes, checkpoint and source provenance,
request lifecycles, action chunks, tick decisions, inference overlap,
per-episode rows, and aggregate values without rerunning inference. The
transport archive SHA-256 is recorded in `transport_sha256.txt`.

## Claim boundary

This is MuJoCo/LIBERO evidence for one attested checkpoint, service, task
matrix, and joint random seed. It is not an official LIBERO leaderboard score,
a universal deadline threshold, a hard-real-time guarantee, hardware safety
evidence, a cross-model comparison, or real-robot deployment.
