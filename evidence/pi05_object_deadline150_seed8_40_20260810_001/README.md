# pi0.5-LIBERO Object 150 ms seed-8 endpoint

This artifact is the registered final seed-8 cell in
[`pi05_object_deadline_transfer_protocol_20260810.json`](../../docs/research/pi05_object_deadline_transfer_protocol_20260810.json).
The seed-8 order was 175 then 150 ms to counterbalance the seed-7 endpoint
order. LIBERO Object tasks 0-9, episodes 0-3, checkpoint, 20 Hz control, H=10
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
| Task success | 0 / 40 (0.0%) |
| Episodes with inference/simulation overlap | 40 / 40 |
| Control ticks / ticks during inference | 11,200 / 11,058 |
| Execute / hold ticks | 2,958 / 8,242 |
| Execute duty cycle | 26.4% |
| Tick-level deadline holds | 8,122 |
| Response-level deadline rejections / provider failures | 1 / 0 |

Tick-level deadline holds and response-level rejections are different units:
one stale response can cause several 20 Hz hold decisions.

## Independent validation

```powershell
& '.\.venv\Scripts\python.exe' -m integrations.openpi.validate_libero_independent_clock `
  '.\evidence\pi05_object_deadline150_seed8_40_20260810_001\evaluation' --json
```

The validator recomputes manifest hashes, provenance, sampling controls,
request lifecycles, action chunks, tick decisions, overlap, and aggregate
outcomes. The verified transport archive hash is in `transport_sha256.txt`.

## Claim boundary

This is exploratory MuJoCo/LIBERO evidence for one checkpoint and service. It
is not a universal deadline threshold, official leaderboard score,
hard-real-time guarantee, hardware safety result, cross-model comparison, or
iid deployment estimate.
