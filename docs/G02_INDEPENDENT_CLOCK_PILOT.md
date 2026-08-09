# G02 independent-clock pi0.5-LIBERO pilot

G02 is the first ArmBench run in which the official `pi05_libero` checkpoint,
the LIBERO simulator, and the runtime scheduler are exercised together with
independent clocks. The simulator owns the 20 Hz control loop in its parent
process. A spawned worker performs blocking OpenPI inference through the
already-attested server. A one-slot latest-only mailbox prevents the control
clock from waiting for inference and makes every supersession, response age,
deadline decision, and hold/execute action inspectable.

## Frozen matrix

| Field | Value |
| --- | --- |
| Suite | LIBERO Spatial |
| Tasks | 0-9 |
| Episodes | 0-3 per task |
| Rollouts | 40 |
| Seed | 7 |
| Control period | 50 ms (20 Hz) |
| Deadline | 200 ms |
| Action horizon | 10 x 7 |
| Policy checkpoint | official `pi05_libero`, attested content SHA-256 `9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5` |

## Result

The run completed all 40 assigned rollouts. LIBERO reported 38 task successes
and two task-4 `max_ticks` failures, for 95.0% on this pilot matrix. The
runtime recorded 4,623 control ticks; 4,521 occurred while the inference worker
was active. Of all ticks, 4,031 executed an action suffix and 592 held the
current gripper command while waiting for a usable response. No response
exceeded the 200 ms deadline and no provider failure was recorded.

These numbers answer an execution-systems question: can an attested action-chunk
policy be evaluated while simulation continues on its own clock, with a
fail-closed deadline policy and complete provenance? They do not establish a
leaderboard score, method superiority, hard-real-time scheduling, hardware
safety, or real-robot deployment. The two failure videos are intentionally
preserved.

## Artifacts and validation

- [Core artifact](../evidence/pi05_libero_independent_clock_core_40_001/README.md)
- [Visual success artifact](../evidence/pi05_libero_independent_clock_visual_success_001/README.md)
- [Evidence catalog](EVIDENCE_CATALOG.md)

Validate either artifact without a GPU:

```bash
python -m integrations.openpi.validate_libero_independent_clock \
  evidence/pi05_libero_independent_clock_core_40_001/evaluation --json
```

The validator does not call the checkpoint. It recomputes manifest hashes,
source snapshots, attested model identity, initial-state digests, action chunk
hashes, stale-prefix selection, request/tick ordering, aggregate statistics,
and video coverage.

## Reproduction boundary

Re-running G02 requires Linux, NVIDIA CUDA, LIBERO, the OpenPI server, and the
public checkpoint. A CPU-only checkout can still validate the saved artifacts,
inspect the videos, and run the independent-clock synthetic smoke in
`INDEPENDENT_CLOCK_RUNTIME.md`.
