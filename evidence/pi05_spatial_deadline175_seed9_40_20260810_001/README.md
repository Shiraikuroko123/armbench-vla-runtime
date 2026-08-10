# pi0.5-LIBERO Spatial 175 ms seed-9 endpoint

This artifact is the registered first seed-9 endpoint in
[`pi05_deadline_followup_protocol_20260810.json`](../../docs/research/pi05_deadline_followup_protocol_20260810.json).
The seed-9 order was 175, 155, then 150 ms to counterbalance the earlier
endpoint order. Tasks 0-9, episodes 0-3, checkpoint, 20 Hz control, H=10
chunks, mailbox, and fail-closed hold semantics remain fixed.

## Provenance

- Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`
- Checkpoint content SHA-256: `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- ArmBench runtime commit: `8686490355c54d1dff9523be0c881d14ab45cda8`

## Result

| Measure | Value |
| --- | ---: |
| Completed rollouts | 40 / 40 |
| Task success | 39 / 40 (97.5%) |
| Episodes with inference/simulation overlap | 40 / 40 |
| Control ticks / ticks during inference | 4,579 / 4,485 |
| Execute / hold ticks | 3,973 / 606 |
| Execute duty cycle | 86.8% |
| Tick-level deadline holds | 486 |
| Response-level deadline rejections / provider failures | 0 / 0 |

## Independent validation

```powershell
& '.\.venv\Scripts\python.exe' -m integrations.openpi.validate_libero_independent_clock `
  '.\evidence\pi05_spatial_deadline175_seed9_40_20260810_001\evaluation' --json
```

The verified transport archive SHA-256 is in `transport_sha256.txt`.

## Claim boundary

This is exploratory MuJoCo/LIBERO evidence for one checkpoint and service. It
is not a universal threshold, official leaderboard score, hard-real-time or
hardware-safety result, or cross-model comparison.
