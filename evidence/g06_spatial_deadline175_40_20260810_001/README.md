# G06: pi0.5-LIBERO 175 ms deadline stress

G06 is the second exploratory intermediate point in the deadline sweep. It
keeps the G02 Spatial task matrix, seed, checkpoint, 20 Hz control clock, and
latest-only independent-clock runtime fixed while setting the fail-closed
response deadline to 175 ms.

## Result

| Measure | Value |
| --- | ---: |
| Completed rollouts | 40 / 40 |
| LIBERO task success | 38 / 40 (95.0%) |
| Episodes with measured inference overlap | 40 / 40 |
| Control ticks | 4,555 |
| Ticks during inference | 4,472 |
| Execute / hold ticks | 3,942 / 613 |
| Deadline-exceeded / provider failures | 0 / 0 |

Relative to G05 at 150 ms, the additional response budget raises the execute
duty cycle from 26.2% to 86.5% and restores 38/40 task successes in this fixed
matrix. This is a useful engineering transition point, not evidence of a
universal 175 ms requirement or a statistically powered threshold estimate.

## Independent validation

From the repository root on Windows:

```powershell
& '.\.venv\Scripts\python.exe' -m integrations.openpi.validate_libero_independent_clock `
  '.\evidence\g06_spatial_deadline175_40_20260810_001\evaluation' --json
```

The validator recomputes provenance, manifests, request lifecycles, action
chunks, control ticks, overlap, per-episode outcomes, and aggregate values
without rerunning inference. The transport archive SHA-256 is recorded in
`transport_sha256.txt`.

## Claim boundary

G06 is MuJoCo/LIBERO simulation evidence for one attested checkpoint, service,
seed, and task matrix. It is exploratory and must not be presented as an
official LIBERO leaderboard score, a hard-real-time guarantee, a hardware
safety result, a cross-model comparison, or a real-robot deployment.
