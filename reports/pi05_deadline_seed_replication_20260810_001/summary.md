# pi0.5-LIBERO deadline seed replication (exploratory)

This report compares two joint environment and keyed-policy sampling seeds for
the same official `pi05_libero` checkpoint, LIBERO Spatial task matrix, 20 Hz
independent-clock runtime, latest-only mailbox, and fail-closed hold rule. The
seed-8 cells were selected after observing seed 7, so this is an exploratory
replication rather than a confirmatory study.

| Seed | Deadline | Task success (Wilson 95% CI) | Execute duty | Tick-level deadline holds | Response-level deadline rejections |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 150 ms | 0 / 40 (0.0%, 0.0-8.8%) | 26.2% | 6,371 | 16 |
| 7 | 175 ms | 38 / 40 (95.0%, 83.5-98.6%) | 86.5% | 495 | 0 |
| 8 | 150 ms | 0 / 40 (0.0%, 0.0-8.8%) | 26.6% | 6,338 | 2 |
| 8 | 175 ms | 37 / 40 (92.5%, 80.1-97.4%) | 86.7% | 501 | 0 |

The paired success-rate differences are +95.0 percentage points for seed 7 and
+92.5 points for seed 8. The execute-duty differences are +60.3 and +60.1
points. Both seeds therefore reproduce the same operational direction.

The mechanism is a runtime clock interaction, not a model-quality threshold:
many responses accepted for one 50 ms control interval at 150 ms become stale
at the next tick, while the 175 ms budget retains that suffix for another tick.
Tick-level deadline holds and response-level deadline rejections are reported
separately because one response can affect multiple control ticks.

The 40 cells contain four episodes from each of ten heterogeneous tasks. They
are not treated as 40 iid draws from a deployment population, and the two seeds
are not pooled into a universal threshold estimate. These results remain
simulation evidence for one checkpoint, service, and task suite; they are not
an official leaderboard score, hard-real-time guarantee, hardware safety
result, cross-model comparison, or real-robot deployment.

## Source artifacts

- [Seed 7, 150 ms](../../evidence/g05_spatial_deadline150_40_20260810_001/README.md)
- [Seed 7, 175 ms](../../evidence/g06_spatial_deadline175_40_20260810_001/README.md)
- [Seed 8, 150 ms](../../evidence/pi05_libero_spatial_deadline150_seed8_40_20260810_001/README.md)
- [Seed 8, 175 ms](../../evidence/pi05_libero_spatial_deadline175_seed8_40_20260810_001/README.md)
