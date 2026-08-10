# G04: pi0.5-LIBERO 50 ms deadline stress control

This artifact is a registered stress condition for the G02 independent-clock
protocol. It keeps the official checkpoint, task matrix, seed, control period,
and environment fixed while reducing the response deadline from 200 ms to
50 ms.

## Frozen protocol

- Suite: libero_spatial, tasks 0-9, episode indices 0-3
- Rollouts: 40, seed 7
- Control clock: 20 Hz (50 ms period)
- Deadline: 50 ms
- Deadline disposition: fail-closed hold
- Policy checkpoint: gs://openpi-assets/checkpoints/pi05_libero
- OpenPI commit: 15a9616a00943ada6c20a0f158e3adb39df2ccac

## Result

| Measure | Value |
| --- | ---: |
| Completed rollouts | 40 / 40 |
| LIBERO task success | 0 / 40 (0.0%) |
| Episodes with measured inference overlap | 40 / 40 |
| Control ticks | 8,800 |
| Ticks during inference | 8,688 |
| Execute / hold ticks | 0 / 8,800 |
| Deadline-exceeded / provider failures | 5,474 / 0 |

Every control tick held instead of executing an expired action. This is an
intentional fail-closed stress result, not a model-quality failure or a dropped
run. All 40 traces and failure videos remain in the artifact.

## Independent validation

From the repository root, run:

    .\.venv\Scripts\python.exe -m integrations.openpi.validate_libero_independent_clock evidence/g04_spatial_deadline50_40_20260810_001/evaluation --json

The validator recomputes the stored protocol, timing, hold/execute decisions,
request lifecycle, overlap proof, and aggregate. The transport archive SHA-256
was 873e85ec8b3d4eaea0b93716c0f9392ac86de711f282cda2d7af3558b4978047.

## Interpretation and boundary

Compared with G02's 200 ms condition, this run demonstrates the operational
trade-off of an overly tight deadline: safety-preserving hold dominates and task
progress goes to zero. It does not prove a universal deadline threshold, an OS
hard-real-time guarantee, hardware safety, or superiority over another VLA.
