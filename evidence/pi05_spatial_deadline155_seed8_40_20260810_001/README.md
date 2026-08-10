# pi0.5-LIBERO Spatial 155 ms seed-8 boundary probe

This artifact is the registered seed-8 155 ms cell in
[`pi05_deadline_followup_protocol_20260810.json`](../../docs/research/pi05_deadline_followup_protocol_20260810.json).
It evaluates tasks 0-9 and episodes 0-3 with the official `pi05_libero`
checkpoint, 20 Hz control, H=10 chunks, a latest-only mailbox, and fail-closed
holds. The joint environment and keyed-policy seed is 8.

## Provenance

- Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`
- Checkpoint content SHA-256: `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5`
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- ArmBench runtime commit: `8686490355c54d1dff9523be0c881d14ab45cda8`

## Result

| Measure | Value |
| --- | ---: |
| Completed rollouts | 40 / 40 |
| Task success | 36 / 40 (90.0%) |
| Episodes with inference/simulation overlap | 40 / 40 |
| Control ticks / ticks during inference | 4,825 / 4,735 |
| Execute / hold ticks | 4,213 / 612 |
| Execute duty cycle | 87.3% |
| Tick-level deadline holds | 492 |
| Response-level deadline rejections / provider failures | 2 / 0 |

## Independent validation

```powershell
& '.\.venv\Scripts\python.exe' -m integrations.openpi.validate_libero_independent_clock `
  '.\evidence\pi05_spatial_deadline155_seed8_40_20260810_001\evaluation' --json
```

The validator recomputes the raw timing and action decisions. The verified
transport archive hash is in `transport_sha256.txt`.

## Claim boundary

This exploratory cell jointly changes environment and policy randomness. It is
not a universal threshold, official leaderboard score, hard-real-time or
hardware-safety result, cross-model comparison, or iid deployment estimate.
