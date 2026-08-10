# pi0.5-LIBERO Spatial 200 ms seed-9 reference

This artifact is the registered seed-9 nominal reference in
[`pi05_deadline_nominal_reference_protocol_20260810.json`](../../docs/research/pi05_deadline_nominal_reference_protocol_20260810.json).
It holds the checkpoint, tasks 0-9, episodes 0-3, 20 Hz control, H=10 chunks,
latest-only mailbox, and fail-closed hold semantics fixed while using the
service-specific 200 ms response deadline.

## Provenance

- Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`
- Checkpoint content SHA-256: `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- ArmBench runtime commit: `8686490355c54d1dff9523be0c881d14ab45cda8`

## Result

| Measure | Value |
| --- | ---: |
| Completed rollouts | 40 / 40 |
| Task success | 38 / 40 (95.0%) |
| Episodes with inference/simulation overlap | 40 / 40 |
| Control ticks / ticks during inference | 4,744 / 4,655 |
| Execute / hold ticks | 4,121 / 623 |
| Execute duty cycle | 86.9% |
| Response-level deadline rejections / provider failures | 0 / 0 |

## Independent validation

```powershell
& '.\.venv\Scripts\python.exe' -m integrations.openpi.validate_libero_independent_clock `
  '.\evidence\pi05_spatial_deadline200_seed9_40_20260810_001\evaluation' --json
```

The verified transport archive SHA-256 is in `transport_sha256.txt`.

## Claim boundary

This is an exploratory matched-seed reference for one checkpoint and service,
not a no-deadline condition, universal threshold, official leaderboard score,
hard-real-time or hardware-safety result, or cross-model comparison.
