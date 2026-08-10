# pi0.5-LIBERO Spatial 155 ms seed-9 boundary probe

This artifact is the registered seed-9 155 ms cell in
[`pi05_deadline_followup_protocol_20260810.json`](../../docs/research/pi05_deadline_followup_protocol_20260810.json).
It evaluates tasks 0-9 and episodes 0-3 with the official `pi05_libero`
checkpoint, 20 Hz control, H=10 chunks, a latest-only mailbox, and fail-closed
holds.

## Provenance

- Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`
- Checkpoint content SHA-256: `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- ArmBench runtime commit: `8686490355c54d1dff9523be0c881d14ab45cda8`

## Result

| Measure | Value |
| --- | ---: |
| Completed rollouts | 40 / 40 |
| Task success | 40 / 40 (100.0%) |
| Episodes with inference/simulation overlap | 40 / 40 |
| Control ticks / ticks during inference | 4,532 / 4,441 |
| Execute / hold ticks | 3,880 / 652 |
| Execute duty cycle | 85.6% |
| Tick-level deadline holds | 532 |
| Response-level deadline rejections / provider failures | 2 / 0 |

## Independent validation

```powershell
& '.\.venv\Scripts\python.exe' -m integrations.openpi.validate_libero_independent_clock `
  '.\evidence\pi05_spatial_deadline155_seed9_40_20260810_001\evaluation' --json
```

The verified transport archive SHA-256 is in `transport_sha256.txt`.

## Claim boundary

This is exploratory MuJoCo/LIBERO evidence. It does not establish a universal
deadline threshold, official leaderboard score, hard-real-time behavior,
hardware safety, cross-model superiority, or an iid deployment rate.
