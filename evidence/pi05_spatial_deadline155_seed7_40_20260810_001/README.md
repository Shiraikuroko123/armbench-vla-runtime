# pi0.5-LIBERO Spatial 155 ms seed-7 boundary probe

This artifact is the registered seed-7 155 ms cell in
[`pi05_deadline_followup_protocol_20260810.json`](../../docs/research/pi05_deadline_followup_protocol_20260810.json).
It keeps the official `pi05_libero` checkpoint, LIBERO Spatial tasks 0-9 and
episodes 0-3, 20 Hz controller, H=10 chunks, latest-only mailbox, and
fail-closed hold rule fixed.

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
| Control ticks / ticks during inference | 4,624 / 4,541 |
| Execute / hold ticks | 4,015 / 609 |
| Execute duty cycle | 86.8% |
| Tick-level deadline holds | 489 |
| Response-level deadline rejections / provider failures | 2 / 0 |

The 155 ms cell restores execution duty relative to 150 ms and is similar to
the registered 175 ms endpoint. This is consistent with a service-clock
quantization transition, not evidence for a universal VLA threshold.

## Independent validation

```powershell
& '.\.venv\Scripts\python.exe' -m integrations.openpi.validate_libero_independent_clock `
  '.\evidence\pi05_spatial_deadline155_seed7_40_20260810_001\evaluation' --json
```

The transport archive SHA-256 is recorded in `transport_sha256.txt`.

## Claim boundary

This is exploratory MuJoCo/LIBERO evidence for one checkpoint and service. It
is not an official leaderboard score, hard-real-time guarantee, hardware
safety result, cross-model comparison, or iid deployment estimate.
