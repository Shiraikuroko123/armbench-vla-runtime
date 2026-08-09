# Independent-clock runtime smoke

This module is the runtime-level experiment behind the project's VLA systems
claim. The parent process owns the environment and advances it on a periodic
wall-clock schedule while a spawned child process performs a potentially
blocking policy call. A one-slot mailbox keeps only the newest pending
observation; every replacement, response age, deadline decision, and process
ID is retained in the returned artifact.

The slot is not implemented as `multiprocessing.Queue(maxsize=1)`. Python's
queue feeder can report an item as absent while its capacity semaphore still
reports full, with different timing on Windows and Linux. ArmBench instead uses
a manager-backed shared slot guarded by a process lock and wake event. The
parent atomically overwrites the one pending object, and the child atomically
removes it before invoking the provider. This keeps observation memory bounded
and preserves latest-only behavior without making control-side submission wait
for policy latency.

Run it on a CPU-only checkout:

```bash
cd project
python -m armbench vla-independent-clock-smoke \
  --policy-latency-ms 160 \
  --control-period-ms 10 \
  --action-period-ms 66.6667 \
  --deadline-ms 200 \
  --max-ticks 20 > independent_clock_smoke.json
```

The command must report `"passed": true`. The evidence demonstrates that
control ticks continue while inference is in flight (`parent_process_id` and
`worker_process_id` differ), that stale pending requests are superseded, and
that late or failed responses fail closed to the configured hold action.

This is a scheduler and provenance smoke, not a learned-policy benchmark. The
fake provider does not establish pi0.5 task success, physical-robot safety, or
operating-system hard-real-time guarantees. The live OpenPI/Panda bridge uses
the same latest-only and deadline concepts, but requires the separately
attested server run described in `LIVE_PI05_PANDA_BRIDGE.md`.
