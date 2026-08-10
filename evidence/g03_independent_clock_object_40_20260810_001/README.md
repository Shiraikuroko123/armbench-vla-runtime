# G03: pi0.5-LIBERO Object independent-clock extension

This artifact extends the G02 independent-clock pilot from libero_spatial to
libero_object using the same attested pi05_libero checkpoint and seed. It is an
exploratory cross-suite validation, not an official LIBERO leaderboard entry.

## Frozen protocol

- Suite: libero_object, tasks 0-9, episode indices 0-3
- Rollouts: 40, seed 7
- Control clock: 20 Hz (50 ms period)
- Inference: separate blocking worker and latest-only mailbox
- Deadline disposition: hold and preserve the last gripper command
- Deadline: 200 ms
- Policy checkpoint: gs://openpi-assets/checkpoints/pi05_libero
- OpenPI commit: 15a9616a00943ada6c20a0f158e3adb39df2ccac

## Result

| Measure | Value |
| --- | ---: |
| Completed rollouts | 40 / 40 |
| LIBERO task success | 39 / 40 (97.5%) |
| Episodes with measured inference overlap | 40 / 40 |
| Control ticks | 6,263 |
| Ticks during inference | 6,161 |
| Execute / hold ticks | 5,530 / 733 |
| Deadline-exceeded / provider failures | 0 / 0 |

The one unsuccessful rollout is retained in evaluation/per_episode.csv and its
full runtime trace. No episode was removed from the denominator.

## Independent validation

From the repository root, run:

    .\.venv\Scripts\python.exe -m integrations.openpi.validate_libero_independent_clock evidence/g03_independent_clock_object_40_20260810_001/evaluation --json

The validator recomputes manifest hashes, source and checkpoint provenance,
request lifecycles, action chunks, tick ordering, overlap, per-episode rows, and
aggregate values. The transport archive SHA-256 was
a6aa5f890351b46201f716798643dd4d0bebb8d34bbfabb69c51ccd43c2e4696.

## Claim boundary

This is a simulation pilot showing that the same runtime protocol transfers to
a second LIBERO suite. It does not establish a complete-suite benchmark score,
method superiority, hard-real-time scheduling, hardware safety, or robot
deployment. The result must not be pooled with G02 as if it were a single frozen
statistical study.
