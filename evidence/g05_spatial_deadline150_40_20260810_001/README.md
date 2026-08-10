# G05: pi0.5-LIBERO 150 ms deadline stress

G05 keeps the G02 LIBERO Spatial matrix, checkpoint, seed, control period, and
independent-clock runtime fixed while reducing the response deadline from
200 ms to 150 ms. It is an exploratory stress point selected after G04, not a
prospectively registered confirmatory comparison.

## Protocol

- Suite: `libero_spatial`, tasks 0-9, episode indices 0-3
- Rollouts: 40, seed 7
- Control clock: 20 Hz (50 ms period)
- Deadline: 150 ms
- Deadline disposition: fail-closed hold
- Inference: separate blocking worker with a latest-only mailbox
- Policy checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`

## Result

| Measure | Value |
| --- | ---: |
| Completed rollouts | 40 / 40 |
| LIBERO task success | 0 / 40 (0.0%) |
| Episodes with measured inference overlap | 40 / 40 |
| Control ticks | 8,800 |
| Ticks during inference | 8,686 |
| Execute / hold ticks | 2,309 / 6,491 |
| Deadline-exceeded / provider failures | 16 / 0 |

Unlike the 50 ms G04 condition, G05 did execute fresh action suffixes on 26.2%
of control ticks. That duty cycle was still insufficient for any rollout to
reach the LIBERO success predicate within its task budget. All unsuccessful
rollouts and videos remain in the artifact.

## Independent validation

From the repository root on Windows:

```powershell
& '.\.venv\Scripts\python.exe' -m integrations.openpi.validate_libero_independent_clock `
  '.\evidence\g05_spatial_deadline150_40_20260810_001\evaluation' --json
```

The validator recomputes manifest hashes, checkpoint and source provenance,
request lifecycles, action chunks, tick decisions, inference overlap,
per-episode rows, and aggregate values without rerunning the model. The
transport archive SHA-256 is recorded in `transport_sha256.txt`.

## Claim boundary

This result demonstrates a bounded engineering trade-off for one checkpoint,
GPU service, simulator suite, seed, and 40-rollout matrix. It does not establish
a universal deadline threshold, an OS hard-real-time guarantee, model-quality
superiority, hardware safety, or real-robot deployment.
